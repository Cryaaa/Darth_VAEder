"""
Plate-level normalization for Cell Painting images.

Statistics are a property of (plate, channel), NOT of a single image. So we fit
ONCE over the whole plate (per channel), save the stats, then apply the same
(center, scale) to every image at load time -- like sklearn fit/transform. This
is why it lives outside the Dataset: computing stats inside __getitem__ would
give per-image stats, not plate-level ones.

Pipeline order:  raw uint16  ->  (illumination correction)  ->  robust-Z
Illumination (flat-field) correction must run BEFORE fitting, because it changes
the pixel distribution -- fit on the same images the model will actually see.

Usage
-----
# 1) fit once (offline), writes plate_norm_BR00149208.json
import pandas as pd
from darth_vaeder.datamodules.normalization import fit_plate_stats
df = pd.read_csv("image_metadata_BR00149208.csv")
fit_plate_stats(df, out_path="plate_norm_BR00149208.json")

# 2) apply inside the Dataset
from darth_vaeder.datamodules.normalization import PlateNormalizer
norm = PlateNormalizer("plate_norm_BR00149208.json")
img = norm(tiff.imread(path), channel)   # float32, ~zero-centered, unit-ish scale
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile as tiff


def _illum_correct(img, illum):
    """Apply BaSiC flat/dark-field correction if provided, else just cast to float.

    illum is a (flatfield, darkfield) tuple of arrays matching the image shape.
    """
    img = img.astype(np.float32)
    if illum is None:
        return img
    flatfield, darkfield = illum
    return (img - darkfield) / np.maximum(flatfield, 1e-6)


def fit_plate_stats(
    df,
    *,
    channel_col="Channel Index",
    path_col="Path",
    method="robust",          # "robust" -> median / IQR ; "zscore" -> mean / std
    sample_per_image=20_000,  # pixels sampled per image (keeps memory tiny)
    illum=None,               # {channel: (flatfield, darkfield)} or None
    seed=0,
    out_path="plate_norm.json",
):
    """Compute one (center, scale) per channel over the WHOLE plate and save them.

    We sample `sample_per_image` random pixels per image and pool them per
    channel, so percentiles (median/IQR) are accurate without loading the full
    ~27 GB plate into memory.
    """
    rng = np.random.default_rng(seed)
    illum = illum or {}
    stats = {}

    for ch, sub in df.groupby(channel_col):
        pool = []
        for p in sub[path_col]:
            img = _illum_correct(tiff.imread(p), illum.get(int(ch)))
            flat = img.ravel()
            if sample_per_image and flat.size > sample_per_image:
                flat = flat[rng.integers(0, flat.size, size=sample_per_image)]
            pool.append(flat.astype(np.float32))
        pool = np.concatenate(pool)

        if method == "robust":
            q1, med, q3 = np.percentile(pool, [25, 50, 75])
            center, scale = float(med), float(max(q3 - q1, 1e-6))
        elif method == "zscore":
            center, scale = float(pool.mean()), float(max(pool.std(), 1e-6))
        else:
            raise ValueError(f"unknown method {method!r}")

        stats[int(ch)] = {
            "method": method,
            "center": center,
            "scale": scale,
            "p01": float(np.percentile(pool, 1)),
            "p99": float(np.percentile(pool, 99)),
            "n_images": int(len(sub)),
            "n_pixels": int(pool.size),
        }
        print(f"ch{int(ch)}: center={center:.1f} scale={scale:.1f} "
              f"(from {len(sub)} images)")

    Path(out_path).write_text(json.dumps(stats, indent=2))
    print(f"wrote {out_path}")
    return stats


class PlateNormalizer:
    """Load fitted plate stats and apply (center, scale) per channel at load time."""

    def __init__(self, stats_path, illum=None):
        raw = json.loads(Path(stats_path).read_text())
        self.stats = {int(k): v for k, v in raw.items()}
        self.illum = illum or {}

    def __call__(self, img, channel):
        s = self.stats[int(channel)]
        img = _illum_correct(img, self.illum.get(int(channel)))
        return (img - s["center"]) / s["scale"]
