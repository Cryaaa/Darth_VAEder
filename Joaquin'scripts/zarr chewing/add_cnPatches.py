"""Build cnPatches + pNucmask in the zarr and save normalization stats to cell_table.csv.

cnPatches is identical to cPatches except the nuclei channel (ch1) is zeroed outside a
5-px dilation of the binary union of cNucmask.  This constrains nuclear signal to the
actual nuclear footprint, improving reconstruction contrast.

pNucmask is that dilated binary mask (disk-5), stored as int32 for consistency with
cCellmask / pCellmask.

Normalization stats (p1/p99) are computed once here:
    norm_mem_lo / norm_mem_hi  — membrane channel, in-mask pixels = pCellmask > 0
    norm_nuc_lo / norm_nuc_hi  — nuclei  channel, in-mask pixels = pNucmask  > 0
                                  (falls back to membrane stats if pNucmask is empty)

Stats are merged into outputs/cell_table.csv (backed up first as cell_table.backup.csv).

Run on the server:
    conda activate darth-vaeder
    python "Joaquin'scripts/add_cnPatches.py" \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table /mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/outputs/cell_table.csv \
        [--dilation 5] [--workers 8] [--dry-run]
"""

import argparse
import concurrent.futures
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.ndimage import binary_dilation
from skimage.morphology import disk


# ── Per-group worker ──────────────────────────────────────────────────────────

def process_group(args):
    """Add pNucmask + cnPatches to one patch group and return per-cell norm stats.

    Returns list of (rep, cond, img, local_cell_index, mem_lo, mem_hi, nuc_lo, nuc_hi).
    """
    zarr_path, group_path, dilation, dry_run = args
    rep, cond, img = group_path.split("/")[1:]

    root = zarr.open_group(zarr_path, mode="r" if dry_run else "r+")
    pg   = root[group_path]

    src_patches  = pg["cPatches"]   # (N, 256, 256, 2)  float32
    src_nucmask  = pg["cNucmask"]   # (N, 256, 256)      int32
    # pCellmask fallback: shouldn't happen in practice (was already added)
    src_cellmask = pg["pCellmask"] if "pCellmask" in pg else pg["cCellmask"]

    N = src_patches.shape[0]
    disk_se = disk(dilation)

    if dry_run:
        return [(rep, cond, img, i, 0., 1., 0., 1.) for i in range(N)]

    # preallocate output arrays (overwrite if re-running) — zarr v2 API
    pg.empty("pNucmask",  shape=src_nucmask.shape,  chunks=src_nucmask.chunks,
             dtype=np.int32,   overwrite=True)
    pg.empty("cnPatches", shape=src_patches.shape,  chunks=src_patches.chunks,
             dtype=np.float32, overwrite=True)
    dst_pnucmask  = pg["pNucmask"]
    dst_cnpatches = pg["cnPatches"]

    stats = []
    qs = torch.tensor([0.01, 0.99])

    for i in range(N):
        patch    = np.ascontiguousarray(src_patches[i])    # (256, 256, 2)
        nucmask  = np.ascontiguousarray(src_nucmask[i])    # (256, 256)  multi-label
        cellmask = np.ascontiguousarray(src_cellmask[i])   # (256, 256)

        # pNucmask: dilate the BINARY UNION of all nuclei (not per-label)
        nuc_binary = nucmask > 0
        if nuc_binary.any():
            pnuc = binary_dilation(nuc_binary, structure=disk_se).astype(np.int32)
        else:
            pnuc = nuc_binary.astype(np.int32)
        dst_pnucmask[i] = pnuc

        # cnPatches: membrane unchanged, nuclei carved to pNucmask
        cn = patch.copy()
        cn[..., 1] = patch[..., 1] * (pnuc > 0)
        dst_cnpatches[i] = cn

        # normalization stats (match NormalizeMasked: p1/p99, in-mask only, no clamp)
        cell_pixels = torch.from_numpy(patch[cellmask > 0, 0])   # membrane in pCellmask
        if cell_pixels.numel() > 1:
            lo, hi = torch.quantile(cell_pixels.float(), qs)
            mem_lo, mem_hi = float(lo), float(hi)
        else:
            mem_lo, mem_hi = 0.0, 1.0

        nuc_pixels = torch.from_numpy(cn[pnuc > 0, 1])           # nuclei in pNucmask
        if nuc_pixels.numel() > 1:
            lo, hi = torch.quantile(nuc_pixels.float(), qs)
            nuc_lo, nuc_hi = float(lo), float(hi)
        else:
            nuc_lo, nuc_hi = mem_lo, mem_hi                       # fallback

        stats.append((rep, cond, img, i, mem_lo, mem_hi, nuc_lo, nuc_hi))

    return stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_groups(zarr_path: str) -> list[str]:
    root = zarr.open_group(zarr_path, mode="r")
    groups = []
    for rep in root["patches"].keys():
        for cond in root["patches"][rep].keys():
            for img in root["patches"][rep][cond].keys():
                groups.append(f"patches/{rep}/{cond}/{img}")
    return groups


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr",     required=True)
    ap.add_argument("--table",    required=True, help="Path to cell_table.csv")
    ap.add_argument("--dilation", type=int, default=5)
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    zarr_path  = str(args.zarr)
    table_path = Path(args.table)

    groups = collect_groups(zarr_path)
    print(f"Found {len(groups)} patch groups.")

    if args.dry_run:
        root  = zarr.open_group(zarr_path, mode="r")
        total = sum(root[g]["cPatches"].shape[0] for g in groups)
        print(f"Dry run: would process {total:,} cells across {len(groups)} groups.")
        return

    # build reverse lookup: (rep, cond, img, local_cell_index) -> cell_idx
    table = pd.read_csv(table_path)
    rev   = {
        (str(r["replicate"]), str(r["condition"]),
         str(r["image_name"]), int(r["local_cell_index"])): int(r["cell_idx"])
        for _, r in table.iterrows()
    }

    work = [(zarr_path, g, args.dilation, False) for g in groups]

    t0         = time.time()
    all_stats  = []
    n_done     = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_group, w): w[1] for w in work}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            rows = fut.result()
            all_stats.extend(rows)
            n_done += len(rows)
            if i % 10 == 0 or i == len(groups):
                print(f"  [{i}/{len(groups)}] {n_done:,} cells done — {time.time()-t0:.1f}s")

    # map (rep, cond, img, loc) -> cell_idx and build stat rows
    stat_rows = []
    missing   = 0
    for rep, cond, img, loc, mem_lo, mem_hi, nuc_lo, nuc_hi in all_stats:
        key = (rep, cond, img, loc)
        if key not in rev:
            missing += 1
            continue
        stat_rows.append({
            "cell_idx":    rev[key],
            "norm_mem_lo": mem_lo,
            "norm_mem_hi": mem_hi,
            "norm_nuc_lo": nuc_lo,
            "norm_nuc_hi": nuc_hi,
        })

    if missing:
        print(f"  [warn] {missing} cells had no matching row in cell_table.csv")

    # backup + merge stats into cell_table.csv
    backup = table_path.with_name("cell_table.backup.csv")
    shutil.copy(table_path, backup)
    print(f"  Backed up table → {backup}")

    stat_df = pd.DataFrame(stat_rows).set_index("cell_idx")
    for col in ["norm_mem_lo", "norm_mem_hi", "norm_nuc_lo", "norm_nuc_hi"]:
        table[col] = table["cell_idx"].map(stat_df[col])

    n_nan = table["norm_mem_lo"].isna().sum()
    if n_nan:
        print(f"  [warn] {n_nan} rows have NaN stats — check for missing zarr entries")

    table.to_csv(table_path, index=False)
    print(f"  Wrote updated table → {table_path}  "
          f"({len(stat_rows):,} cells with norm stats)")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
