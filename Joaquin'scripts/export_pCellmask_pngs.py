"""Export pCellmask patches as PNGs organised by condition for ShapeEmbedLite.

Output structure:
    <out>/
        CTRL/
            {cell_idx}_{rep}_{image_name}.png
        MATURE/
            {cell_idx}_{rep}_{image_name}.png
        CMs25d/
            {cell_idx}_{rep}_{image_name}.png

Each PNG is an 8-bit grayscale binary mask: 255 = cell, 0 = background.

Usage
-----
    python "Joaquin'scripts/export_pCellmask_pngs.py" \
        --zarr   /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table  outputs/cell_table.csv \
        --out    /mnt/efs/dl_jrc/student_data/S-JS/pCellmasks \
        --workers 16
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import zarr
from PIL import Image


def _export_cell(args):
    """Worker: load one cell's pCellmask and save as PNG. Returns (cell_idx, ok, msg)."""
    zarr_path, rep, cond, img, local_idx, cell_idx, out_path = args
    try:
        root = zarr.open_group(zarr_path, mode="r")
        pg   = root[f"patches/{rep}/{cond}/{img}"]
        src  = "pCellmask" if "pCellmask" in pg else "cCellmask"
        mask = pg[src][int(local_idx)]          # (H, W) int32
        binary = ((mask > 0) * 255).astype(np.uint8)
        Image.fromarray(binary, mode="L").save(out_path)
        return cell_idx, True, ""
    except Exception as e:
        return cell_idx, False, str(e)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",    required=True,  help="Path to multinucleation.zarr")
    p.add_argument("--table",   required=True,  help="Path to cell_table.csv")
    p.add_argument("--out",     default="/mnt/efs/dl_jrc/student_data/S-JS/pCellmasks",
                                help="Root output folder")
    p.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    args = p.parse_args()

    out_root = Path(args.out)
    df = pd.read_csv(args.table)
    print(f"  {len(df)} cells to export")

    # Create one subfolder per condition
    conditions = df["condition"].unique()
    for cond in conditions:
        (out_root / cond).mkdir(parents=True, exist_ok=True)
    print(f"  conditions: {sorted(conditions)}")

    # Build work list
    work = []
    for _, row in df.iterrows():
        rep      = str(row["replicate"])
        cond     = str(row["condition"])
        img      = str(row["image_name"])
        local    = int(row["local_cell_index"])
        cell_idx = int(row["cell_idx"])
        fname    = f"{cell_idx}_{rep}_{img}.png"
        out_path = str(out_root / cond / fname)
        work.append((args.zarr, rep, cond, img, local, cell_idx, out_path))

    print(f"  starting export with {args.workers} workers …")
    n_ok = n_fail = 0
    with Pool(processes=args.workers) as pool:
        for i, (cell_idx, ok, msg) in enumerate(pool.imap_unordered(_export_cell, work, chunksize=32)):
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  ERROR cell {cell_idx}: {msg}")
            if (i + 1) % 1000 == 0 or (i + 1) == len(work):
                print(f"  {i+1}/{len(work)}  ok={n_ok}  fail={n_fail}", end="\r")

    print(f"\n  done — {n_ok} exported, {n_fail} failed")
    print(f"  output: {out_root}")


if __name__ == "__main__":
    main()
