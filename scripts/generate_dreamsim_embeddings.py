#!/usr/bin/env python3
"""
Step 1a — Generate DreamSim embeddings for vision-guided alignment.

For 3D CT volumes (``--modality ct``) the encoder is applied to
``--num-slices`` center slices extracted with a center-of-mass crop.  Each
slice embedding is stored separately so that the alignment step can perform
slice-aligned cosine similarity.

For 2D images such as chest X-rays (``--modality cxr``) the encoder is
applied directly to the full image and a single embedding is stored.

In both cases each sample is saved as:

    {output_dir}/{sample_id}.pt

The .pt file contains a list of dicts:
    [{"embedding": Tensor(D,), "slice_idx": int}, ...]

For 2D images the list has exactly one element with ``slice_idx=0``.

Resumable: samples whose .pt file already exists are skipped.

Usage
-----
# Abdominal CT (3D) — CT preprocessing uses the fixed transforms_image pipeline
python scripts/generate_dreamsim_embeddings.py \\
    --metadata /data/pool_metadata.csv \\
    --image-dir /data/ct_scans/ \\
    --output-dir /data/embeddings/dreamsim/ \\
    --modality ct \\
    --id-column sample_id

# Chest X-ray (2D)
python scripts/generate_dreamsim_embeddings.py \\
    --metadata /data/pool_metadata.csv \\
    --image-dir /data/cxr_images/ \\
    --output-dir /data/embeddings/dreamsim/ \\
    --modality cxr \\
    --id-column sample_id
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg_curation.embeddings.dreamsim import (
    compute_dreamsim_embedding_2d,
    compute_dreamsim_embeddings_ct,
    setup_dreamsim_model,
)
from rg_curation.utils.ct_preprocessing import extract_center_slices, get_ct_transforms
from rg_curation.utils.text_utils import get_memory_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DreamSim embeddings for a medical imaging dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="CSV file with at least an ID column and an 'image_file' column.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing the image files referenced in --metadata.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where {sample_id}.pt embedding files will be written.",
    )
    parser.add_argument(
        "--modality",
        choices=["ct", "cxr"],
        default="ct",
        help="Image modality.  'ct' expects .nii.gz volumes; 'cxr' expects 2D images.",
    )
    parser.add_argument(
        "--id-column",
        default="sample_id",
        help="Name of the unique sample identifier column in the metadata CSV.",
    )
    parser.add_argument(
        "--image-column",
        default="image_file",
        help="Name of the image filename column in the metadata CSV.",
    )
    # CT-specific slice extraction
    parser.add_argument(
        "--num-slices",
        type=int,
        default=10,
        help="[CT only] Number of center slices to extract per volume.",
    )
    parser.add_argument(
        "--slice-spacing",
        type=int,
        default=2,
        help="[CT only] Spacing between consecutive extracted slices.",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.85,
        help="[CT only] Center-of-mass crop ratio (0 < ratio < 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.metadata)
    logger.info(f"Loaded metadata: {len(df)} records from {args.metadata}")

    if args.id_column not in df.columns:
        logger.error(f"ID column '{args.id_column}' not found.  Available: {df.columns.tolist()}")
        sys.exit(1)
    if args.image_column not in df.columns:
        logger.error(f"Image column '{args.image_column}' not found.  Available: {df.columns.tolist()}")
        sys.exit(1)

    # Skip samples that already have embeddings
    existing = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(args.output_dir, "*.pt"))
    }
    before = len(df)
    df = df[~df[args.id_column].isin(existing)]
    logger.info(f"Skipping {before - len(df)} samples (already processed).  "
                f"Remaining: {len(df)}")

    if df.empty:
        logger.info("All samples already processed.")
        return

    model, preprocess, device = setup_dreamsim_model()

    if args.modality == "ct":
        transforms = get_ct_transforms()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

    processed = errors = 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="DreamSim embeddings")):
        sample_id = row[args.id_column]
        img_basename = os.path.basename(row[args.image_column])
        img_path = os.path.join(args.image_dir, img_basename)

        if not os.path.isfile(img_path):
            logger.warning(f"Image not found: {img_path}")
            errors += 1
            continue

        try:
            if args.modality == "ct":
                ct_data = transforms({"image": img_path})
                volume = ct_data["image"]
                cropped_slices, slice_indices = extract_center_slices(
                    volume,
                    num_slices=args.num_slices,
                    slice_spacing=args.slice_spacing,
                    crop_ratio=args.crop_ratio,
                )
                if len(cropped_slices) == 0:
                    logger.warning(f"No slices extracted for {sample_id}")
                    errors += 1
                    continue
                embs = compute_dreamsim_embeddings_ct(
                    cropped_slices, model, preprocess, device
                )
                slice_data = [
                    {"embedding": embs[j], "slice_idx": slice_indices[j]}
                    for j in range(len(slice_indices))
                ]
            else:  # cxr
                emb = compute_dreamsim_embedding_2d(img_path, model, preprocess, device)
                slice_data = [{"embedding": emb[0], "slice_idx": 0}]

            pt_path = os.path.join(args.output_dir, f"{sample_id}.pt")
            torch.save(slice_data, pt_path)
            processed += 1

        except Exception as exc:
            logger.error(f"Error processing {sample_id}: {exc}")
            errors += 1
            continue

        if device.type == "cuda" and i % 20 == 0:
            torch.cuda.empty_cache()

        if processed % 500 == 0 and processed > 0:
            logger.info(f"Processed {processed} samples | RAM: {get_memory_usage():.0f} MB")

    logger.info(
        f"Done — processed: {processed}, errors/skipped: {errors}, "
        f"total attempted: {len(df)}"
    )
    logger.info(f"Embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
