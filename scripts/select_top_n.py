#!/usr/bin/env python3
"""
Step 5 — Select the top-N samples from any scored CSV.

A thin utility that takes any CSV containing a score column and writes the
top-N rows (sorted descending) to a new file.  Works with the output of:

  - run_alignment.py      → sort by ``similarity_score``
  - run_rrf.py            → sort by ``rrf_score``
  - compute_clip_scores.py → sort by ``similarity_score``

Usage
-----
python scripts/select_top_n.py \\
    --input  /data/rrf_k5.csv \\
    --output /data/curated_10k.csv \\
    --n      10000 \\
    --score-column rrf_score
"""

import argparse
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a scored CSV to the top-N samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Input CSV with a score column.")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for the top-N samples.")
    parser.add_argument("--n", type=int, required=True,
                        help="Number of samples to keep.")
    parser.add_argument(
        "--score-column",
        default="similarity_score",
        help="Name of the column to sort by (descending).",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        default=False,
        help="Sort ascending instead of descending (e.g. for distance-based scores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} samples from: {args.input}")

    if args.score_column not in df.columns:
        logger.error(
            f"Score column '{args.score_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
        sys.exit(1)

    df_sorted = df.sort_values(args.score_column, ascending=args.ascending)

    if args.n >= len(df_sorted):
        logger.warning(
            f"Requested top-{args.n} but only {len(df_sorted)} samples available. "
            "Keeping all."
        )
        top_df = df_sorted
    else:
        top_df = df_sorted.head(args.n)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    top_df.to_csv(args.output, index=False)

    score_col = top_df[args.score_column]
    logger.info(f"Saved {len(top_df)} samples to: {args.output}")
    logger.info(
        f"Score range — min: {score_col.min():.4f}, "
        f"mean: {score_col.mean():.4f}, "
        f"max: {score_col.max():.4f}"
    )


if __name__ == "__main__":
    main()
