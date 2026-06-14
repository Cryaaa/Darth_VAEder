"""Add pCellmask to every patch group in the zarr store.

pCellmask is cCellmask dilated by DILATION_PX pixels using a circular
structuring element.  It matches the mask that was used to crop cPatches
(which was slightly larger than the tight cell boundary), allowing
NormalizeMasked to compute percentile statistics over the correct region.

Run on the server:
    python scripts/add_pCellmask.py \
        --zarr /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        [--dilation 5] [--workers 8] [--dry-run]
"""

import argparse
import concurrent.futures
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure
from skimage.morphology import disk


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary-dilate a 2-D label mask by px pixels (circular footprint).

    Non-zero pixels are treated as foreground.  The original label value is
    preserved in dilated pixels.  Background stays 0.
    """
    if mask.max() == 0:
        return mask.copy()
    label = int(mask.max())          # single cell per chunk — one non-zero label
    fg    = mask > 0
    dilated = binary_dilation(fg, structure=disk(px))
    return (dilated * label).astype(mask.dtype)


def process_group(args):
    """Worker: add pCellmask to one patch group.  Returns (path_str, n_cells)."""
    zarr_path, group_path, dilation, dry_run = args
    root = zarr.open_group(zarr_path, mode="r+" if not dry_run else "r")
    pg   = root[group_path]

    src = pg["cCellmask"]   # (n, H, W) int32
    n   = src.shape[0]

    if "pCellmask" in pg and not dry_run:
        # Already exists — verify shape matches and skip
        if pg["pCellmask"].shape == src.shape:
            return group_path, 0   # 0 = skipped
        else:
            del pg["pCellmask"]    # shape mismatch → recreate

    if dry_run:
        return group_path, n

    dst = pg.require_dataset(
        "pCellmask",
        shape=src.shape,
        chunks=src.chunks,
        dtype=src.dtype,
        overwrite=False,
    )

    for i in range(n):
        dst[i] = dilate_mask(src[i], dilation)

    return group_path, n


def collect_groups(zarr_path: str) -> list[str]:
    """Walk the zarr tree and return all patch group paths (leaf level)."""
    root   = zarr.open_group(zarr_path, mode="r")
    groups = []
    for rep in root["patches"].keys():
        for cond in root["patches"][rep].keys():
            for img in root["patches"][rep][cond].keys():
                groups.append(f"patches/{rep}/{cond}/{img}")
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr",     required=True, help="Path to multinucleation.zarr")
    ap.add_argument("--dilation", type=int, default=5,
                    help="Dilation radius in pixels (default 5)")
    ap.add_argument("--workers",  type=int, default=8,
                    help="Parallel workers (default 8)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="List groups and cell counts without writing")
    args = ap.parse_args()

    zarr_path = str(args.zarr)
    groups    = collect_groups(zarr_path)
    print(f"Found {len(groups)} patch groups.")

    if args.dry_run:
        total = sum(zarr.open_group(zarr_path, mode="r")[g]["cCellmask"].shape[0]
                    for g in groups)
        print(f"Dry run: would dilate {total:,} cell masks by {args.dilation} px.")
        return

    work = [(zarr_path, g, args.dilation, False) for g in groups]

    t0         = time.time()
    total_done = 0
    skipped    = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_group, w): w[1] for w in work}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            gpath, n = fut.result()
            if n == 0:
                skipped += 1
            else:
                total_done += n
            if i % 20 == 0 or i == len(groups):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(groups)}] {total_done:,} cells written, "
                      f"{skipped} groups skipped — {elapsed:.1f}s")

    print(f"\nDone.  {total_done:,} cells dilated by {args.dilation} px "
          f"({skipped} groups already had pCellmask).  "
          f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
