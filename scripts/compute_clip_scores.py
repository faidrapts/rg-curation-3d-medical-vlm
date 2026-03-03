#!/usr/bin/env python3
"""
Step 2 (CLIP baseline) — Compute per-sample CLIP alignment scores.

For each scan–report pair in the metadata CSV the script computes the cosine
similarity between the CLIP vision embedding and the CLIP text embedding of
the findings section.  This score is used as a baseline filtering criterion
(higher score ≈ better image–text alignment).

For 3D CT volumes (``--modality ct``) the vision embedding is the
L2-normalised mean of per-slice CLIP embeddings extracted using the same
slice configuration as the DreamSim step.

For 2D images (``--modality cxr``) a single CLIP vision embedding is computed
directly from the image.

Output is a CSV with columns:
    {id_column}, image_file, findings, similarity_score

Usage
-----
# CT
python scripts/compute_clip_scores.py \\
    --metadata /data/pool_metadata.csv \\
    --image-dir /data/ct_scans/ \\
    --output /data/clip_scores.csv \\
    --modality ct

# CXR
python scripts/compute_clip_scores.py \\
    --metadata /data/pool_metadata.csv \\
    --image-dir /data/cxr_images/ \\
    --output /data/clip_scores.csv \\
    --modality cxr
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.embeddings.clip_score import (
    compute_clip_score_2d,
    compute_clip_scores_batch,
    setup_clip_model,
)
from rg_curation.utils.ct_preprocessing import extract_center_slices, get_ct_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-sample CLIP alignment scores (image–report cosine similarity)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True,
                        help="CSV with ID, image_file, and findings columns.")
    parser.add_argument("--image-dir", required=True,
                        help="Directory containing image files.")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for CLIP scores.")
    parser.add_argument("--modality", choices=["ct", "cxr"], default="ct")
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--image-column", default="image_file")
    parser.add_argument("--text-column", default="findings")
    # CLIP model
    parser.add_argument("--clip-model", default="ViT-B-32",
                        help="OpenCLIP architecture (e.g. ViT-B-32, ViT-L-14).")
    parser.add_argument("--clip-pretrained", default="datacomp_xl_s13b_b90k",
                        help="OpenCLIP pretrained weights tag.")
    # CT-specific
    parser.add_argument("--num-slices", type=int, default=10)
    parser.add_argument("--slice-spacing", type=int, default=2)
    parser.add_argument("--crop-ratio", type=float, default=0.85)
    # Batching
    parser.add_argument("--batch-scans", type=int, default=32,
                        help="[CT] Number of scans loaded per iteration.")
    parser.add_argument("--vision-batch-size", type=int, default=64,
                        help="Number of image slices per GPU forward pass.")
    parser.add_argument("--load-workers", type=int, default=4,
                        help="[CT] Parallel threads for I/O / MONAI transforms.")
    return parser.parse_args()


def _load_ct_volume(args_tuple):
    """Thread-safe CT volume loader.  Returns (cropped_slices, row, error_msg)."""
    img_path, row, transforms, num_slices, slice_spacing, crop_ratio = args_tuple
    try:
        ct_data = transforms({"image": img_path})
        volume = ct_data["image"]
        cropped_slices, _ = extract_center_slices(
            volume, num_slices=num_slices, slice_spacing=slice_spacing, crop_ratio=crop_ratio
        )
        return cropped_slices, row, None
    except Exception as exc:
        return None, row, str(exc)


def main() -> None:
    args = parse_args()
    np.random.seed(42)

    df = pd.read_csv(args.metadata)
    logger.info(f"Loaded metadata: {len(df)} records")

    for col in (args.id_column, args.image_column, args.text_column):
        if col not in df.columns:
            logger.error(f"Column '{col}' not found. Available: {df.columns.tolist()}")
            sys.exit(1)

    model, preprocess, tokenizer, device = setup_clip_model(
        model_name=args.clip_model, pretrained=args.clip_pretrained
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    results = []
    errors = 0
    n_total = len(df)

    if args.modality == "cxr":
        for _, row in tqdm(df.iterrows(), total=n_total, desc="CLIP scores (CXR)"):
            img_path = os.path.join(args.image_dir, os.path.basename(row[args.image_column]))
            if not os.path.isfile(img_path):
                errors += 1
                continue
            try:
                score = compute_clip_score_2d(
                    img_path,
                    row.get(args.text_column, ""),
                    model, preprocess, tokenizer, device,
                )
                results.append({
                    args.id_column: row[args.id_column],
                    args.image_column: row[args.image_column],
                    args.text_column: row.get(args.text_column, ""),
                    "similarity_score": score,
                })
            except Exception as exc:
                logger.warning(f"Error for {row[args.id_column]}: {exc}")
                errors += 1

    else:  # CT
        transforms = get_ct_transforms()

        def iter_chunks():
            for start in range(0, n_total, args.batch_scans):
                yield df.iloc[start : start + args.batch_scans]

        n_chunks = (n_total + args.batch_scans - 1) // args.batch_scans
        for chunk_df in tqdm(iter_chunks(), total=n_chunks, desc="CLIP scores (CT)"):
            load_args = []
            for _, row in chunk_df.iterrows():
                img_path = os.path.join(
                    args.image_dir, os.path.basename(row[args.image_column])
                )
                if os.path.isfile(img_path):
                    load_args.append((
                        img_path, row, transforms,
                        args.num_slices, args.slice_spacing, args.crop_ratio,
                    ))
                else:
                    errors += 1

            if not load_args:
                continue

            slices_list = []
            rows_ok = []
            with ThreadPoolExecutor(max_workers=args.load_workers) as executor:
                futures = {executor.submit(_load_ct_volume, a): a for a in load_args}
                for future in as_completed(futures):
                    slices, row, err = future.result()
                    if err:
                        logger.warning(f"Load error ({row[args.id_column]}): {err}")
                        errors += 1
                    else:
                        slices_list.append(slices)
                        rows_ok.append(row)

            if not slices_list:
                continue

            try:
                scores = compute_clip_scores_batch(
                    slices_list,
                    [r.get(args.text_column, "") for r in rows_ok],
                    model, preprocess, tokenizer, device,
                    vision_batch_size=args.vision_batch_size,
                )
            except Exception as exc:
                logger.warning(f"Batch CLIP error: {exc}")
                errors += len(rows_ok)
                continue

            for row, score in zip(rows_ok, scores):
                results.append({
                    args.id_column: row[args.id_column],
                    args.image_column: row[args.image_column],
                    args.text_column: row.get(args.text_column, ""),
                    "similarity_score": score,
                })

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not results:
        logger.error("No results produced.")
        sys.exit(1)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(out_df)} CLIP scores to: {args.output}")
    logger.info(
        f"Similarity — mean: {out_df['similarity_score'].mean():.4f}, "
        f"min: {out_df['similarity_score'].min():.4f}, "
        f"max: {out_df['similarity_score'].max():.4f}"
    )
    if errors:
        logger.warning(f"Skipped / errors: {errors}")


if __name__ == "__main__":
    main()
