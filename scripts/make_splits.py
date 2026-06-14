"""Assign cells to train / val / test splits and save them as JSON.

Splits are made at the IMAGE level (default) so that all cells from the same
field of view land in the same split, preventing data leakage. The saved JSON
maps each split to a list of global ``ncells_idx`` and can be shared so everyone
trains/evaluates on the exact same partition.

Usage
-----
    python scripts/make_splits.py \
        --zarr /path/to/multinucleation.zarr \
        --out outputs/splits.json \
        --ratios 0.7 0.15 0.15 \
        --split-by image --stratify-by condition --seed 42
"""

import argparse
import json

import numpy as np

from darth_vaeder.datamodules import build_splits, load_cell_index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", required=True, help="Path to multinucleation.zarr")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--split-by", choices=["image", "cell"], default="image")
    p.add_argument("--stratify-by", default="condition",
                   help="Categorical column to balance across splits, or 'none'")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stratify = None if args.stratify_by.lower() == "none" else args.stratify_by
    routing = load_cell_index(args.zarr)
    splits = build_splits(routing, ratios=tuple(args.ratios), split_by=args.split_by,
                          stratify_by=stratify, seed=args.seed)

    with open(args.out, "w") as f:
        json.dump({k: np.asarray(v).tolist() for k, v in splits.items()}, f)

    total = sum(len(v) for v in splits.values())
    print(f"Wrote {args.out}")
    for k, v in splits.items():
        print(f"  {k:5s}: {len(v):6d} cells  ({len(v) / total:5.1%})")
    print(f"  total: {total:6d} cells")


if __name__ == "__main__":
    main()
