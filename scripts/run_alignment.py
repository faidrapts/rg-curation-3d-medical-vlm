#!/usr/bin/env python3
"""
Step 3 — Reference-guided kNN alignment to filter a training pool.

Retrieves the k most similar samples from a training pool (D0) for each
sample in a small reference set (D_ref) using pre-computed embeddings.
The pool is then deduplicated: each sample is retained at most once, with
a ``similarity_score`` equal to its best match across all reference queries.

Three alignment methods are supported:

  vision_only    — DreamSim slice embeddings (slice-aligned cosine similarity)
  text_only      — Clinical Longformer sentence embeddings (cosine similarity)
  early_fusion   — Concatenated DreamSim + Longformer (slice-aligned cosine
                   similarity on the joint representation)

Output is a CSV containing the retained pool samples together with:
  - similarity_score
  - ref_{id_column}   (closest reference match)
  - neighbor_rank     (1 = closest match to any reference sample)

Usage
-----
# Vision-only alignment
python scripts/run_alignment.py \\
    --pool-metadata  /data/pool.csv \\
    --ref-metadata   /data/reference.csv \\
    --method         vision_only \\
    --vision-embeddings-dir /data/embeddings/dreamsim/ \\
    --k 5 \\
    --output /data/aligned_vision_k5.csv

# Text-only alignment
python scripts/run_alignment.py \\
    --pool-metadata  /data/pool.csv \\
    --ref-metadata   /data/reference.csv \\
    --method         text_only \\
    --text-embeddings-dir /data/embeddings/longformer/ \\
    --k 5 \\
    --output /data/aligned_text_k5.csv

# Early fusion
python scripts/run_alignment.py \\
    --pool-metadata  /data/pool.csv \\
    --ref-metadata   /data/reference.csv \\
    --method         early_fusion \\
    --vision-embeddings-dir /data/embeddings/dreamsim/ \\
    --text-embeddings-dir   /data/embeddings/longformer/ \\
    --k 5 \\
    --output /data/aligned_early_fusion_k5.csv

Note
----
When the pool and reference sets share the same embedding directory (the
common case) you only need to provide a single --vision-embeddings-dir /
--text-embeddings-dir.  If they differ, use --pool-vision-embeddings-dir /
--pool-text-embeddings-dir to override the pool-specific path.
"""

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.filtering.alignment import run_knn_alignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-guided kNN alignment for training pool curation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pool-metadata", required=True,
                        help="CSV for the uncurated training pool (D0).")
    parser.add_argument("--ref-metadata", required=True,
                        help="CSV for the reference set (D_ref).")
    parser.add_argument(
        "--method",
        required=True,
        choices=["vision_only", "text_only", "early_fusion"],
        help="Alignment method.",
    )
    parser.add_argument("--output", required=True,
                        help="Output CSV path for the aligned pool subset.")
    parser.add_argument("--id-column", default="sample_id",
                        help="Unique sample identifier column in both CSVs.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of nearest neighbours per reference sample.")
    # Embedding directories
    parser.add_argument(
        "--vision-embeddings-dir",
        default=None,
        help="Directory with DreamSim .pt files.  Used for both pool and "
             "reference unless overridden by --pool-vision-embeddings-dir.",
    )
    parser.add_argument(
        "--text-embeddings-dir",
        default=None,
        help="Directory with Longformer .pt files.  Used for both pool and "
             "reference unless overridden by --pool-text-embeddings-dir.",
    )
    parser.add_argument(
        "--pool-vision-embeddings-dir",
        default=None,
        help="[Optional] Override vision embedding directory for the pool.",
    )
    parser.add_argument(
        "--pool-text-embeddings-dir",
        default=None,
        help="[Optional] Override text embedding directory for the pool.",
    )
    # Top-N selection (optional convenience flag)
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Keep only the top-N samples by similarity score. "
             "Equivalent to running select_top_n.py afterwards.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate that required embedding dirs are provided for the chosen method.
    if args.method in ("vision_only", "early_fusion") and args.vision_embeddings_dir is None:
        logger.error("--vision-embeddings-dir is required for 'vision_only' and 'early_fusion'")
        sys.exit(1)
    if args.method in ("text_only", "early_fusion") and args.text_embeddings_dir is None:
        logger.error("--text-embeddings-dir is required for 'text_only' and 'early_fusion'")
        sys.exit(1)

    pool_df = pd.read_csv(args.pool_metadata)
    ref_df = pd.read_csv(args.ref_metadata)
    logger.info(f"Pool: {len(pool_df)} samples | Reference: {len(ref_df)} samples")

    for df, name in ((pool_df, "pool"), (ref_df, "reference")):
        if args.id_column not in df.columns:
            logger.error(
                f"ID column '{args.id_column}' not found in {name} CSV. "
                f"Available: {df.columns.tolist()}"
            )
            sys.exit(1)

    aligned_df = run_knn_alignment(
        pool_df=pool_df,
        ref_df=ref_df,
        method=args.method,
        k=args.k,
        id_column=args.id_column,
        vision_embeddings_dir=args.vision_embeddings_dir,
        text_embeddings_dir=args.text_embeddings_dir,
        pool_vision_embeddings_dir=args.pool_vision_embeddings_dir,
        pool_text_embeddings_dir=args.pool_text_embeddings_dir,
    )

    if aligned_df.empty:
        logger.error("Alignment produced no results.")
        sys.exit(1)

    if args.top_n is not None and args.top_n < len(aligned_df):
        aligned_df = aligned_df.head(args.top_n).copy()
        logger.info(f"Filtered to top {args.top_n} samples by similarity score.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    aligned_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(aligned_df)} aligned samples to: {args.output}")


if __name__ == "__main__":
    main()
