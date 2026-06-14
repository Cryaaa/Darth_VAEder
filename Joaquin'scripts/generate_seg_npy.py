"""
Reconstruct CellPose-compatible _seg.npy files for every upsampled image.

The seg.npy format is a pickled dict that the CellPose GUI reads to restore
a segmentation session.  We reconstruct every field the GUI needs:

  masks          (H, W) uint16  – upsampled cp_masks
  outlines       (H, W) uint16  – boundary pixels labelled with their cell ID
  colors         (N, 3) uint8   – one random RGB colour per cell
  filename       str            – absolute path to the upsampled image TIFF
  flows          [[], [], [], [], []]  – empty (GUI reconstructs on load)
  ismanual       (N,) bool      – all False (no manual corrections yet)
  manual_changes []             – empty history
  model_path     0
  flow_threshold      0.4
  cellprob_threshold  0.0
  normalize_params    dict      – CellPose defaults
  restore        None
  ratio          1.0
  diameter       None
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from cellpose import utils   # for masks_to_outlines


def make_seg_npy(masks: np.ndarray, img_path: Path) -> dict:
    """
    Build the seg.npy dict from a (H, W) uint16 mask array.
    """
    n_cells = int(masks.max())

    # outlines: boundary pixels carry the label value (not just True/False)
    outline_bool = utils.masks_to_outlines(masks)          # (H, W) bool
    outlines     = (masks * outline_bool).astype(np.uint16)

    # one random colour per cell, reproducible per image via label seed
    rng    = np.random.default_rng(seed=hash(img_path.name) % (2**32))
    colors = rng.integers(50, 230, size=(n_cells, 3), dtype=np.uint8)

    return {
        "masks":   masks.astype(np.uint16),
        "outlines": outlines,
        "colors":  colors,
        "filename": str(img_path.resolve()),
        "flows":   [[], [], [], [], []],
        "ismanual": np.zeros(n_cells, dtype=bool),
        "manual_changes": [],
        "model_path":  0,
        "flow_threshold":     0.4,
        "cellprob_threshold": 0.0,
        "normalize_params": {
            "lowhigh": None, "percentile": [1.0, 99.0],
            "normalize": True, "norm3D": True,
            "sharpen_radius": 0.0, "smooth_radius": 0.0,
            "tile_norm_blocksize": 0.0, "tile_norm_smooth3D": 0.0,
            "invert": False,
        },
        "restore":  None,
        "ratio":    1.0,
        "diameter": None,
    }


def generate_all(data_upsampled: Path, meta_path: Path):
    meta = pd.read_csv(meta_path)
    ok, missing = 0, []

    for _, row in meta.iterrows():
        sample = row["sample"]
        stem   = Path(row["filename"]).stem

        img_path  = data_upsampled / sample / row["filename"]
        mask_path = data_upsampled / sample / f"{stem}_cp_masks.tiff"
        seg_dst   = data_upsampled / sample / f"{stem}_seg.npy"

        if not mask_path.exists():
            print(f"  [WARN] missing cp_masks: {sample}/{stem}")
            missing.append(f"{sample}/{stem}")
            continue

        masks = tifffile.imread(mask_path)
        dat   = make_seg_npy(masks, img_path)
        np.save(seg_dst, dat, allow_pickle=True)
        print(f"  ✓  {sample}/{stem}  cells={int(masks.max())}  → {seg_dst.name}")
        ok += 1

    print(f"\nGenerated : {ok}")
    if missing:
        print(f"Skipped   : {len(missing)}")
        for m in missing:
            print(f"  ✗  {m}")


if __name__ == "__main__":
    BASE      = Path("/Users/joaco/Documents/Janelia/Multinucleation Big")
    OUT_DIR   = BASE / "data_upsampled"
    META_PATH = BASE / "multinucleation_image_metadata.csv"
    generate_all(OUT_DIR, META_PATH)
