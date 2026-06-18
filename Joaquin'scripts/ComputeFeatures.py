"""Compute geometric and texture features for all cells in multinucleation.zarr
and write them as new columns into cell_table.csv.

Geometric features (pixel units)
---------------------------------
Computed on pCellmask and pNucmask independently via skimage regionprops.
Columns prefixed gFeat_cell_* and gFeat_nuc_*.
12 metrics (unchanged from the original script):
  n_pixels, area, perimeter, circularity, solidity, convex_area,
  equivalent_diameter, eccentricity, major_axis_length, minor_axis_length,
  orientation, extent.

Note: pNucmask is the binary UNION of all nuclei (dilated 5px), so for
multinucleated cells the geom describes the combined nuclear footprint, not
per-nucleus.

Texture features (5 per channel)
----------------------------------
Computed on cnPatches (ch0 = membrane, ch1 = nuclei carved to pNucmask).
Intensity is scaled by the stored p1/p99 stats (norm_mem_lo/hi, norm_nuc_lo/hi)
so texture lives on the same scale the model sees.

Features: entropy, glcm_contrast, glcm_homogeneity, glcm_correlation, glcm_energy
  - In-mask pixels quantized to levels 1-31 (level 0 = background)
  - GLCM built at distances=[1], angles=[0, π/4, π/2, 3π/4], averaged over angles
  - Row/col 0 zeroed before computing props to exclude background pairs
Columns prefixed tFeat_mem_* (within pCellmask) and tFeat_nuc_* (within pNucmask).

Usage
-----
    # inspect distributions, no writes
    python "Joaquin'scripts/ComputeFeatures.py" \\
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \\
        --table outputs/cell_table.csv \\
        --dry-run

    # write feature columns to cell_table.csv
    python "Joaquin'scripts/ComputeFeatures.py" \\
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \\
        --table outputs/cell_table.csv
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from skimage.measure import label as sk_label, regionprops
from skimage.feature import graycomatrix, graycoprops

# ---------------------------------------------------------------------------
# Geometric features
# ---------------------------------------------------------------------------

_GEOM_KEYS = [
    "n_pixels", "area", "perimeter", "circularity", "solidity", "convex_area",
    "equivalent_diameter", "eccentricity", "major_axis_length",
    "minor_axis_length", "orientation", "extent",
]


def _geom_features(mask_2d: np.ndarray) -> dict:
    """Compute regionprops-based geometric features on a binary 2D mask."""
    n_px = int(mask_2d.sum())
    row = {"n_pixels": n_px}
    if n_px == 0:
        row.update({k: np.nan for k in _GEOM_KEYS if k != "n_pixels"})
        return row

    props = regionprops(mask_2d.astype(np.uint8))[0]
    area = props.area
    perim = props.perimeter
    row["area"] = area
    row["perimeter"] = perim
    row["circularity"] = 4 * np.pi * area / perim ** 2 if perim > 0 else np.nan
    row["solidity"] = props.solidity
    row["convex_area"] = getattr(props, "area_convex", getattr(props, "convex_area", np.nan))
    row["equivalent_diameter"] = getattr(
        props, "equivalent_diameter_area", getattr(props, "equivalent_diameter", np.nan)
    )
    row["eccentricity"] = props.eccentricity
    row["major_axis_length"] = props.major_axis_length
    row["minor_axis_length"] = props.minor_axis_length
    row["orientation"] = props.orientation
    row["extent"] = props.extent
    return row


# ---------------------------------------------------------------------------
# Texture features
# ---------------------------------------------------------------------------

_TEX_KEYS = ["entropy", "glcm_contrast", "glcm_homogeneity", "glcm_correlation", "glcm_energy"]
_GLCM_LEVELS = 32   # 0 = background, 1-31 = in-mask intensities
_GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def _tex_features(img_ch: np.ndarray, mask_2d: np.ndarray, lo: float, hi: float) -> dict:
    """Compute entropy + Haralick GLCM texture features for one channel.

    img_ch  : (H, W) float32 — raw cnPatches channel (may be zero outside mask)
    mask_2d : (H, W) binary — defines the region of interest
    lo, hi  : p1/p99 stats for [0,1] rescaling (same scale as model input)
    """
    row = {k: np.nan for k in _TEX_KEYS}
    px = mask_2d.astype(bool)
    if not px.any():
        return row

    # clip + scale to [0, 1] using stored p1/p99 stats
    vals = img_ch[px].astype(np.float64)
    denom = float(hi) - float(lo)
    if denom > 0:
        vals = (vals - float(lo)) / denom
    vals = np.clip(vals, 0.0, 1.0)

    # Shannon entropy over in-mask pixel histogram (32 bins)
    hist, _ = np.histogram(vals, bins=_GLCM_LEVELS - 1, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum > 0:
        p = hist / hist_sum
        p = p[p > 0]
        row["entropy"] = float(-np.sum(p * np.log2(p)))

    # quantize in-mask pixels to 1..(GLCM_LEVELS-1); background stays 0
    quantized = np.zeros(img_ch.shape, dtype=np.uint8)
    q_vals = np.floor(vals * (_GLCM_LEVELS - 1 - 1)).astype(np.uint8) + 1
    q_vals = np.clip(q_vals, 1, _GLCM_LEVELS - 1)
    quantized[px] = q_vals

    # build GLCM and zero out background row/col before computing props
    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=_GLCM_ANGLES,
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=False,
    )  # shape (levels, levels, 1, n_angles)

    # zero background (level 0) to exclude non-mask pixel pairs
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0

    # renormalize per distance/angle slice so props are well-defined
    col_sum = glcm.sum(axis=(0, 1), keepdims=True)
    col_sum[col_sum == 0] = 1
    glcm_norm = glcm.astype(np.float64) / col_sum

    try:
        row["glcm_contrast"]    = float(graycoprops(glcm_norm, "contrast").mean())
        row["glcm_homogeneity"] = float(graycoprops(glcm_norm, "homogeneity").mean())
        row["glcm_energy"]      = float(graycoprops(glcm_norm, "energy").mean())
        corr = graycoprops(glcm_norm, "correlation")
        # correlation can be NaN when std is 0 (flat patch); replace with 0
        corr = np.where(np.isfinite(corr), corr, 0.0)
        row["glcm_correlation"] = float(corr.mean())
    except Exception:
        pass  # leave NaN on any skimage version edge-case

    return row


# ---------------------------------------------------------------------------
# Per-group worker
# ---------------------------------------------------------------------------

def process_group(args_tuple):
    """Compute all features for every cell in one patch group."""
    zarr_path, rep, cond, img, local_idxs, cell_idxs, norm_rows = args_tuple

    root = zarr.open_group(zarr_path, mode="r")
    pg = root[f"patches/{rep}/{cond}/{img}"]

    pcell = pg["pCellmask"]   # (N, H, W) int32
    pnuc  = pg["pNucmask"]    # (N, H, W) int32
    cnp   = pg["cnPatches"]   # (N, H, W, 2) float32

    results = []
    for loc, ci, nrow in zip(local_idxs, cell_idxs, norm_rows):
        loc = int(loc)
        cell_mask = (pcell[loc] > 0)
        nuc_mask  = (pnuc[loc]  > 0)
        img_mem   = cnp[loc, :, :, 0]
        img_nuc   = cnp[loc, :, :, 1]

        # geometric
        gf_cell = {f"gFeat_cell_{k}": v for k, v in _geom_features(cell_mask).items()}
        gf_nuc  = {f"gFeat_nuc_{k}":  v for k, v in _geom_features(nuc_mask).items()}

        # texture
        tf_mem = {f"tFeat_mem_{k}": v for k, v in _tex_features(
            img_mem, cell_mask, nrow["norm_mem_lo"], nrow["norm_mem_hi"]).items()}
        tf_nuc = {f"tFeat_nuc_{k}": v for k, v in _tex_features(
            img_nuc, nuc_mask, nrow["norm_nuc_lo"], nrow["norm_nuc_hi"]).items()}

        results.append((int(ci), {**gf_cell, **gf_nuc, **tf_mem, **tf_nuc}))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",    required=True, help="Path to multinucleation.zarr")
    p.add_argument("--table",   required=True, help="Path to cell_table.csv")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="Compute features on first 3 groups and print distributions; no writes")
    args = p.parse_args()

    table_path = Path(args.table)
    df = pd.read_csv(table_path)
    print(f"  loaded {len(df)} cells from {table_path}")

    # build per-group work units
    groups: dict = {}
    norm_stat_cols = ["norm_mem_lo", "norm_mem_hi", "norm_nuc_lo", "norm_nuc_hi"]
    for _, row in df.iterrows():
        key = (str(row["replicate"]), str(row["condition"]), str(row["image_name"]))
        groups.setdefault(key, []).append((
            int(row["local_cell_index"]),
            int(row["cell_idx"]),
            {c: float(row[c]) for c in norm_stat_cols},
        ))

    work = [
        (args.zarr, rep, cond, img,
         [t[0] for t in triples],
         [t[1] for t in triples],
         [t[2] for t in triples])
        for (rep, cond, img), triples in groups.items()
    ]

    if args.dry_run:
        work = work[:3]
        print(f"  dry-run: processing {len(work)} groups only")

    print(f"  processing {len(work)} patch groups with {args.workers} workers …")

    ci_to_feats: dict = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_group, w): w for w in work}
        done = 0
        for fut in as_completed(futs):
            for ci, feat_dict in fut.result():
                ci_to_feats[ci] = feat_dict
            done += 1
            if done % 20 == 0 or done == len(work):
                print(f"    {done}/{len(work)} groups done", end="\r")
    print()

    # assemble feature dataframe aligned to df
    feat_rows = [ci_to_feats.get(int(ci), {}) for ci in df["cell_idx"]]
    feat_df = pd.DataFrame(feat_rows)

    if args.dry_run:
        print(f"\n  feature columns ({len(feat_df.columns)}):")
        for col in feat_df.columns:
            vals = feat_df[col].dropna()
            if len(vals):
                print(f"    {col:40s}  min={vals.min():.4g}  median={vals.median():.4g}  max={vals.max():.4g}  nan={feat_df[col].isna().sum()}")
        print("\n  dry-run: no changes written.")
        return

    # backup + merge + write
    backup = table_path.with_name(table_path.stem + ".backup.csv")
    shutil.copy2(table_path, backup)
    print(f"  backed up: {backup}")

    # drop any existing feature columns to allow re-runs
    existing_feat_cols = [c for c in df.columns if c.startswith(("gFeat_", "tFeat_"))]
    if existing_feat_cols:
        df = df.drop(columns=existing_feat_cols)
        print(f"  dropped {len(existing_feat_cols)} existing feature columns (re-run)")

    result = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    result.to_csv(table_path, index=False)
    print(f"  wrote {len(feat_df.columns)} feature columns → {table_path}")
    print(f"  total columns now: {len(result.columns)}")


if __name__ == "__main__":
    main()
