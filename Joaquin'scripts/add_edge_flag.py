"""Screen the zarr for edge/cropped cells and write edge_run_px into cell_table.csv.

For each cell, computes the longest contiguous run of cCellmask pixels on any of
the 4 patch borders.  Cells with a long run have a flat, straight cut — they are
cropped by the image boundary.  Cells that merely touch the border tangentially
have short runs (1–5 px).

The raw pixel count (edge_run_px) is written to cell_table.csv so you can tune
the threshold at train time without re-running this script.

Usage
-----
    # inspect distribution, no writes
    python "Joaquin'scripts/add_edge_flag.py" \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv \
        --dry-run

    # write edge_run_px column
    python "Joaquin'scripts/add_edge_flag.py" \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import label as nd_label


BORDER_MARGIN = 5  # px band near each edge to check for cell pixels

def max_run_on_border(cell_mask: np.ndarray) -> int:
    """Longest contiguous run of pCellmask pixels inside the border margin band.

    Checks a BORDER_MARGIN-pixel-wide strip along each of the 4 sides rather
    than just the literal edge row/column.  Cells whose pCellmask stops 1-2 px
    inside the border (scoring 0 with a single-row check) are caught here,
    preventing them from forming orientation-based subclusters in the UMAP.
    """
    m = BORDER_MARGIN
    best = 0
    strips = [
        cell_mask[:m,  :].max(axis=0),   # top    band → collapse to 1-D along width
        cell_mask[-m:, :].max(axis=0),   # bottom band
        cell_mask[:,  :m].max(axis=1),   # left   band → collapse along height
        cell_mask[:, -m:].max(axis=1),   # right  band
    ]
    for strip in strips:
        lbl, n = nd_label(strip > 0)
        if n:
            run = max(int((lbl == i).sum()) for i in range(1, n + 1))
            best = max(best, run)
    return best


def process_group(args_tuple):
    """Worker: compute edge_run_px for all cells in one patch group."""
    zarr_path, rep, cond, img, local_idxs, cell_idxs = args_tuple
    root = zarr.open_group(zarr_path, mode="r")
    pg   = root[f"patches/{rep}/{cond}/{img}"]
    # use pCellmask (dilated, matches actual crop boundary) — cCellmask is too
    # eroded to reach the patch border even for visually-cropped cells
    cmask_arr = pg["pCellmask"] if "pCellmask" in pg else pg["cCellmask"]  # (N, H, W)

    results = []
    for loc, ci in zip(local_idxs, cell_idxs):
        cmask = cmask_arr[int(loc)]          # (H, W)
        run   = max_run_on_border(cmask)
        results.append((int(ci), run))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",    required=True)
    p.add_argument("--table",   required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="Print distribution and candidate thresholds; do not write CSV")
    args = p.parse_args()

    table_path = Path(args.table)
    df = pd.read_csv(table_path)
    print(f"  loaded {len(df)} cells from {table_path}")

    # Build per-group work units: {(rep,cond,img) -> [(local_idx, cell_idx), ...]}
    groups = {}
    for _, row in df.iterrows():
        key = (str(row["replicate"]), str(row["condition"]), str(row["image_name"]))
        groups.setdefault(key, []).append((int(row["local_cell_index"]), int(row["cell_idx"])))

    work = [
        (args.zarr, rep, cond, img,
         [t[0] for t in pairs], [t[1] for t in pairs])
        for (rep, cond, img), pairs in groups.items()
    ]
    print(f"  processing {len(work)} patch groups with {args.workers} workers …")

    ci_to_run = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_group, w): w for w in work}
        done = 0
        for fut in as_completed(futs):
            for ci, run in fut.result():
                ci_to_run[ci] = run
            done += 1
            if done % 20 == 0 or done == len(work):
                print(f"    {done}/{len(work)} groups done", end="\r")
    print()

    runs = np.array([ci_to_run[ci] for ci in df["cell_idx"]])
    print(f"\n  edge_run_px distribution:")
    print(f"    min={runs.min()}  p25={int(np.percentile(runs,25))}  "
          f"median={int(np.median(runs))}  p75={int(np.percentile(runs,75))}  max={runs.max()}")
    print(f"\n  cells that WOULD be dropped at each threshold:")
    for thr in [5, 10, 15, 20, 25, 30]:
        n_drop = int((runs >= thr).sum())
        print(f"    threshold >= {thr:>3} px  →  drop {n_drop}/{len(runs)}  "
              f"({100*n_drop/len(runs):.1f}%)")

    if args.dry_run:
        print("\n  dry-run: no changes written.")
        return

    # Back up the original CSV
    backup = table_path.with_name(table_path.stem + ".backup.csv")
    shutil.copy2(table_path, backup)
    print(f"\n  backed up: {backup}")

    df["edge_run_px"] = runs
    df.to_csv(table_path, index=False)
    print(f"  wrote edge_run_px column → {table_path}")


if __name__ == "__main__":
    main()
