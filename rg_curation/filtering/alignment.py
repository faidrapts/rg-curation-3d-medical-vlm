"""
Reference-guided kNN alignment for filtering a training pool.

Given pre-computed embeddings for both a training pool (D0) and a small
high-quality reference set (D_ref), this module finds the k nearest
neighbours in D0 for each reference sample, then deduplicates so that each
pool sample is retained at most once — with the similarity score equal to
its maximum cosine similarity to any reference sample.

Three alignment strategies are supported:

``vision_only``
    Uses DreamSim slice embeddings.  Similarity is computed as the
    average slice-aligned cosine similarity between two volumes.

``text_only``
    Uses Clinical Longformer sentence embeddings (one vector per sample).
    Similarity is standard cosine similarity.

``early_fusion``
    Concatenates per-slice DreamSim embeddings with the (broadcast)
    Longformer embedding, then applies slice-aligned cosine similarity on
    the joint representation.  Each modality is L2-normalised before
    concatenation to ensure equal contribution.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------

def load_dreamsim_embedding(
    sample_id: str,
    embeddings_dir: str,
) -> Optional[np.ndarray]:
    """Load DreamSim slice embeddings for *sample_id* from *embeddings_dir*.

    The .pt file is expected to contain a list of dicts, each with an
    ``"embedding"`` key (as saved by the generation script).

    Args:
        sample_id: Unique sample identifier (used as the filename stem).
        embeddings_dir: Directory containing ``{sample_id}.pt`` files.

    Returns:
        Float32 NumPy array of shape ``(num_slices, D)`` or ``None`` if the
        file is missing or cannot be loaded.
    """
    pt_path = os.path.join(embeddings_dir, f"{sample_id}.pt")
    if not os.path.exists(pt_path):
        return None
    try:
        data = torch.load(pt_path, map_location="cpu")
        embs = np.array([entry["embedding"].numpy() for entry in data], dtype=np.float32)
        return embs
    except Exception as exc:
        logger.warning(f"Could not load DreamSim embedding for {sample_id}: {exc}")
        return None


def load_text_embedding(
    sample_id: str,
    embeddings_dir: str,
) -> Optional[np.ndarray]:
    """Load a Clinical Longformer text embedding for *sample_id*.

    The .pt file is expected to contain a dict with an ``"embedding"`` key
    (as saved by the generation script).

    Args:
        sample_id: Unique sample identifier (filename stem).
        embeddings_dir: Directory containing ``{sample_id}.pt`` files.

    Returns:
        1-D float32 NumPy array or ``None`` if the file is missing or corrupt.
    """
    pt_path = os.path.join(embeddings_dir, f"{sample_id}.pt")
    if not os.path.exists(pt_path):
        return None
    try:
        data = torch.load(pt_path, map_location="cpu")
        return data["embedding"].numpy().astype(np.float32)
    except Exception as exc:
        logger.warning(f"Could not load text embedding for {sample_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Embedding normalisation and similarity
# ---------------------------------------------------------------------------

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Row-normalise an array of embeddings to unit L2 norm.

    Args:
        embeddings: Array of shape ``(N, D)`` or ``(D,)``.

    Returns:
        L2-normalised array of the same shape.
    """
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        return embeddings / max(norm, 1e-12)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def compute_slice_aligned_similarity(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> float:
    """Average slice-aligned cosine similarity between two volumes.

    Slices are compared in order up to the minimum slice count of the two
    volumes.  Both inputs must already be L2-normalised.

    Args:
        emb_a: Array of shape ``(S_a, D)`` — normalised slice embeddings.
        emb_b: Array of shape ``(S_b, D)`` — normalised slice embeddings.

    Returns:
        Mean cosine similarity (float).
    """
    n = min(len(emb_a), len(emb_b))
    if n == 0:
        return 0.0
    return float(np.mean(np.sum(emb_a[:n] * emb_b[:n], axis=1)))


def build_early_fusion_embedding(
    dreamsim_emb: np.ndarray,
    text_emb: np.ndarray,
) -> np.ndarray:
    """Concatenate normalised DreamSim and text embeddings (early fusion).

    The Longformer embedding is broadcast to match the number of DreamSim
    slices so that slice-aligned similarity can be computed on the joint
    representation.

    Args:
        dreamsim_emb: Array of shape ``(S, D_v)`` — DreamSim slice
            embeddings (not necessarily normalised).
        text_emb: 1-D array of shape ``(D_t,)`` — Longformer embedding.

    Returns:
        Float32 array of shape ``(S, D_v + D_t)`` with unit-norm rows.
    """
    dreamsim_norm = normalize_embeddings(dreamsim_emb)
    text_norm = normalize_embeddings(text_emb)
    text_broadcast = np.tile(text_norm, (len(dreamsim_emb), 1))
    fused = np.concatenate([dreamsim_norm, text_broadcast], axis=1).astype(np.float32)
    return normalize_embeddings(fused)


# ---------------------------------------------------------------------------
# k-NN search
# ---------------------------------------------------------------------------

def _find_knn_text(
    query_emb: np.ndarray,
    pool_embs: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch cosine-similarity kNN for single-vector embeddings (text/2D).

    Args:
        query_emb: 1-D normalised query embedding.
        pool_embs: 2-D array of shape ``(N, D)`` — pool embeddings stacked
            as rows.
        k: Number of nearest neighbours.

    Returns:
        Tuple of ``(indices, similarities)`` for the top-k pool items,
        sorted descending by similarity.
    """
    sims = cosine_similarity([query_emb], pool_embs)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    return top_k, sims[top_k]


def _find_knn_slices(
    query_norm: np.ndarray,
    pool_cache: Dict[str, np.ndarray],
    pool_ids: List[str],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice-aligned cosine-similarity kNN for volumetric embeddings.

    Args:
        query_norm: Normalised slice embeddings of shape ``(S_q, D)``.
        pool_cache: Dict mapping sample IDs to normalised slice embedding
            arrays.
        pool_ids: Ordered list of sample IDs (defines index positions).
        k: Number of nearest neighbours.

    Returns:
        Tuple of ``(indices, similarities)`` for the top-k pool items.
    """
    sims = np.array(
        [
            compute_slice_aligned_similarity(query_norm, pool_cache[sid])
            if sid in pool_cache
            else 0.0
            for sid in pool_ids
        ]
    )
    top_k = np.argsort(sims)[-k:][::-1]
    return top_k, sims[top_k]


# ---------------------------------------------------------------------------
# Main alignment entry point
# ---------------------------------------------------------------------------

def run_knn_alignment(
    pool_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    method: str,
    k: int,
    id_column: str = "sample_id",
    vision_embeddings_dir: Optional[str] = None,
    text_embeddings_dir: Optional[str] = None,
    pool_vision_embeddings_dir: Optional[str] = None,
    pool_text_embeddings_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run reference-guided kNN alignment to filter a training pool.

    For each reference sample the function retrieves the *k* most similar
    candidates from the pool.  The pool is then deduplicated: each sample
    is kept once with the similarity score equal to its best match across
    all reference queries.

    Args:
        pool_df: DataFrame for the training pool (D0).  Must contain
            ``id_column``.
        ref_df: DataFrame for the reference set (D_ref).  Must contain
            ``id_column``.
        method: One of ``"vision_only"``, ``"text_only"``,
            ``"early_fusion"``.
        k: Number of nearest neighbours per reference sample.
        id_column: Name of the unique sample identifier column
            (default: ``"sample_id"``).
        vision_embeddings_dir: Directory with DreamSim .pt files for the
            *reference* set.  Falls back to ``pool_vision_embeddings_dir``
            if ``None``.
        text_embeddings_dir: Directory with Longformer .pt files for the
            *reference* set.  Falls back to ``pool_text_embeddings_dir``
            if ``None``.
        pool_vision_embeddings_dir: Directory with DreamSim .pt files for
            the *pool*.  Required for ``"vision_only"`` and
            ``"early_fusion"``.
        pool_text_embeddings_dir: Directory with Longformer .pt files for
            the *pool*.  Required for ``"text_only"`` and
            ``"early_fusion"``.

    Returns:
        Deduplicated DataFrame of aligned pool samples.  Added columns:

        - ``similarity_score``: max cosine similarity to any reference
          sample.
        - ``ref_sample_id``: identifier of the closest reference match.
        - ``neighbor_rank``: rank of this pool sample within its matched
          reference query (1 = closest).
    """
    if method not in ("vision_only", "text_only", "early_fusion"):
        raise ValueError(f"Unknown method '{method}'.  Choose from: "
                         "vision_only, text_only, early_fusion")

    use_vision = method in ("vision_only", "early_fusion")
    use_text = method in ("text_only", "early_fusion")

    # Default: same directory for pool and reference embeddings.
    if vision_embeddings_dir is None:
        vision_embeddings_dir = pool_vision_embeddings_dir
    if text_embeddings_dir is None:
        text_embeddings_dir = pool_text_embeddings_dir

    pool_vision_dir = pool_vision_embeddings_dir or vision_embeddings_dir
    pool_text_dir = pool_text_embeddings_dir or text_embeddings_dir

    pool_ids = pool_df[id_column].tolist()

    # ------------------------------------------------------------------
    # Load pool embeddings into memory
    # ------------------------------------------------------------------
    logger.info("Loading pool embeddings into memory…")
    pool_emb_cache: Dict[str, np.ndarray] = {}
    missing_vision = missing_text = 0

    for _, row in tqdm(pool_df.iterrows(), total=len(pool_df), desc="Pool embeddings"):
        sid = row[id_column]

        if use_vision and use_text:
            v_emb = load_dreamsim_embedding(sid, pool_vision_dir)
            t_emb = load_text_embedding(sid, pool_text_dir)
            if v_emb is None:
                missing_vision += 1
                continue
            if t_emb is None:
                missing_text += 1
                continue
            emb = normalize_embeddings(build_early_fusion_embedding(v_emb, t_emb))
        elif use_vision:
            v_emb = load_dreamsim_embedding(sid, pool_vision_dir)
            if v_emb is None:
                missing_vision += 1
                continue
            emb = normalize_embeddings(v_emb)
        else:  # text only
            t_emb = load_text_embedding(sid, pool_text_dir)
            if t_emb is None:
                missing_text += 1
                continue
            emb = normalize_embeddings(t_emb)

        pool_emb_cache[sid] = emb

    valid_pool_ids = [s for s in pool_ids if s in pool_emb_cache]
    logger.info(
        f"Pool embeddings loaded: {len(pool_emb_cache)} / {len(pool_ids)} "
        f"(missing vision: {missing_vision}, missing text: {missing_text})"
    )

    # For text-only we stack all embeddings into a matrix for efficient batch
    # cosine-similarity computation.
    if method == "text_only":
        pool_matrix = np.vstack([pool_emb_cache[s] for s in valid_pool_ids])
    else:
        pool_matrix = None  # Not used; slice-aligned computation uses cache directly.

    # ------------------------------------------------------------------
    # Process reference set: find k-NN in pool
    # ------------------------------------------------------------------
    logger.info(f"Running {method} kNN alignment (k={k}) over {len(ref_df)} reference samples…")
    aligned_records = []
    skipped_ref = 0

    for _, ref_row in tqdm(ref_df.iterrows(), total=len(ref_df), desc="Reference kNN"):
        ref_id = ref_row[id_column]

        if use_vision and use_text:
            v_emb = load_dreamsim_embedding(ref_id, vision_embeddings_dir)
            t_emb = load_text_embedding(ref_id, text_embeddings_dir)
            if v_emb is None or t_emb is None:
                skipped_ref += 1
                continue
            ref_emb = normalize_embeddings(build_early_fusion_embedding(v_emb, t_emb))
        elif use_vision:
            v_emb = load_dreamsim_embedding(ref_id, vision_embeddings_dir)
            if v_emb is None:
                skipped_ref += 1
                continue
            ref_emb = normalize_embeddings(v_emb)
        else:
            t_emb = load_text_embedding(ref_id, text_embeddings_dir)
            if t_emb is None:
                skipped_ref += 1
                continue
            ref_emb = normalize_embeddings(t_emb)

        try:
            if method == "text_only":
                nn_indices, nn_sims = _find_knn_text(ref_emb, pool_matrix, k)
                nn_ids = [valid_pool_ids[i] for i in nn_indices]
            else:
                nn_indices, nn_sims = _find_knn_slices(
                    ref_emb, pool_emb_cache, valid_pool_ids, k
                )
                nn_ids = [valid_pool_ids[i] for i in nn_indices]
        except Exception as exc:
            logger.error(f"kNN failed for reference sample {ref_id}: {exc}")
            continue

        for rank, (pool_sample_id, sim) in enumerate(zip(nn_ids, nn_sims)):
            aligned_records.append(
                {
                    id_column: pool_sample_id,
                    f"ref_{id_column}": ref_id,
                    "similarity_score": float(sim),
                    "neighbor_rank": rank + 1,
                }
            )

    if skipped_ref:
        logger.warning(
            f"Skipped {skipped_ref} reference samples (missing embeddings)"
        )

    if not aligned_records:
        logger.error("No aligned records produced.")
        return pd.DataFrame()

    aligned_df = pd.DataFrame(aligned_records)
    logger.info(f"Pre-deduplication: {len(aligned_df)} (pool_id, ref_id) pairs")

    # ------------------------------------------------------------------
    # Deduplication: keep each pool sample once (best rank = rank 1)
    # ------------------------------------------------------------------
    aligned_df = aligned_df.sort_values([id_column, "neighbor_rank"])
    aligned_df = aligned_df.drop_duplicates(subset=[id_column], keep="first")
    aligned_df = aligned_df.sort_values("similarity_score", ascending=False)
    aligned_df = aligned_df.reset_index(drop=True)

    # Merge back with pool metadata columns
    pool_meta = pool_df.set_index(id_column)
    aligned_df = aligned_df.join(pool_meta, on=id_column, how="left")

    logger.info(f"Post-deduplication: {len(aligned_df)} unique pool samples retained")
    logger.info(
        f"Similarity score — mean: {aligned_df['similarity_score'].mean():.4f}, "
        f"min: {aligned_df['similarity_score'].min():.4f}, "
        f"max: {aligned_df['similarity_score'].max():.4f}"
    )
    return aligned_df
