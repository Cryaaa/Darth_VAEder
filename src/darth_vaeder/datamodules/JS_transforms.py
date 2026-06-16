"""On-the-fly transforms for the multinucleation VAE.

Design principles
-----------------
* Geometric augmentations delegate to torchvision.transforms.v2, which handles
  the image/mask type dispatch automatically:
    - tv_tensors.Image  → bilinear interpolation
    - tv_tensors.Mask   → nearest-neighbour interpolation
  The same random parameters are shared across all keys in a single call,
  keeping cPatch and cCellmask registered.
* Custom transforms (NormalizeMasked, MaskBackground) stay hand-rolled because
  they operate on the masked-cell domain and have no torchvision equivalent.
* Every transform takes and returns the sample dict.
* image_keys / mask_keys on every transform let you target any subset of the
  sample dict — add future augmentations the same way.
* Normalisation comes BEFORE geometry: stats computed on raw values, then
  geometry interpolates the already-normalised patch.

Pipeline (first run — cPatch + cCellmask only)
----------------------------------------------
    Compose([
        NormalizeMasked(),       # percentile on in-mask pixels, bg stays 0, no clamp
        RandomRotate360(),       # uniform [0, 360) — same angle for image + mask
        RandomHFlip(),           # p=0.5 — same flip for image + mask
        RandomVFlip(),           # p=0.5 — same flip for image + mask
        MaskBackground(),        # re-zero background after bilinear rotation
    ])

To add a torchvision v2 augmentation that targets only cPatch (e.g. Gaussian blur):
    transforms.transforms.append(
        _TVWrapper(T.GaussianBlur(kernel_size=5), image_keys=("cPatch",))
    )
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import Image as TVImage, Mask as TVMask


class Compose:
    """Chain transforms that each accept and return a sample dict."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ── Normalisation ─────────────────────────────────────────────────────────────

class NormalizeMasked:
    """Per-channel percentile normalisation using in-mask pixels only.

    Stats (p_low, p_high) are computed from pixels where mask_key > 0.
    The scale  (x - p_low) / (p_high - p_low)  is applied ONLY to in-mask
    pixels.  Background pixels (mask == 0) are left at exactly 0.

    No clamping — pixels brighter than p_high map to > 1; this lets the
    network see the full dynamic range of outlier pixels.

    Parameters
    ----------
    patch_key   sample-dict key of the image to normalise (default "cPatch")
    mask_key    sample-dict key of the binary mask         (default "cCellmask")
    low, high   percentiles used as the normalisation anchors (default 1 / 99)
    eps         guards against flat channels (dead detectors, fully dark cells)
    """

    def __init__(
        self,
        patch_key: str = "cPatch",
        mask_key:  str = "pCellmask",
        low:  float = 1.0,
        high: float = 99.0,
        eps:  float = 1e-6,
    ):
        self.patch_key = patch_key
        self.mask_key  = mask_key
        self.qs  = torch.tensor([low / 100.0, high / 100.0])
        self.eps = eps

    def __call__(self, sample: dict) -> dict:
        img  = sample[self.patch_key]        # (C, H, W)  float32  raw
        mask = sample[self.mask_key][0] > 0  # (H, W)     bool
        out  = img.clone()                   # copy — background stays 0
        qs   = self.qs.to(img.dtype)

        for c in range(img.shape[0]):
            pixels = img[c][mask]
            if pixels.numel() == 0:
                continue
            lo, hi = torch.quantile(pixels, qs)
            out[c][mask] = (img[c][mask] - lo) / (hi - lo + self.eps)

        sample[self.patch_key] = out
        return sample


# ── torchvision v2 geometric wrapper ─────────────────────────────────────────

class _TVWrapper:
    """Wrap a torchvision v2 geometric transform to work with our sample dicts.

    Annotates image tensors as tv_tensors.Image and mask tensors as
    tv_tensors.Mask, calls the transform (which shares random params across all
    inputs), then strips the annotations back to plain tensors.

    This is the extension point for any future torchvision augmentation.
    """

    def __init__(
        self,
        tv_transform,
        image_keys: tuple = ("cPatch",),
        mask_keys:  tuple = ("pCellmask",),
    ):
        self.tv_transform = tv_transform
        self.image_keys   = tuple(image_keys)
        self.mask_keys    = tuple(mask_keys)

    def __call__(self, sample: dict) -> dict:
        tv_input: dict[str, TVImage | TVMask] = {}
        for k in self.image_keys:
            if k in sample:
                tv_input[k] = TVImage(sample[k].float())
        for k in self.mask_keys:
            if k in sample:
                tv_input[k] = TVMask(sample[k])

        tv_output = self.tv_transform(tv_input)

        for k, v in tv_output.items():
            sample[k] = torch.as_tensor(v)
        return sample


def RandomRotate360(
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("pCellmask",),
) -> _TVWrapper:
    """Uniform random rotation in [0°, 360°) — bilinear for images, nearest for masks."""
    return _TVWrapper(
        T.RandomRotation(degrees=(0, 360), fill=0),
        image_keys, mask_keys,
    )


def RandomHFlip(
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("pCellmask",),
    p: float = 0.5,
) -> _TVWrapper:
    """Random horizontal flip with probability p."""
    return _TVWrapper(T.RandomHorizontalFlip(p=p), image_keys, mask_keys)


def RandomVFlip(
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("pCellmask",),
    p: float = 0.5,
) -> _TVWrapper:
    """Random vertical flip with probability p."""
    return _TVWrapper(T.RandomVerticalFlip(p=p), image_keys, mask_keys)


# ── Background cleanup ────────────────────────────────────────────────────────

class MaskBackground:
    """Zero out image pixels wherever the cell mask is 0.

    Bilinear rotation smears a thin fringe of cell pixels into the background.
    Applying this after all geometric transforms restores the invariant that
    background == 0 without affecting in-mask content.

    Parameters
    ----------
    image_keys  keys to zero out in background regions
    mask_key    binary mask that defines foreground (> 0 = cell)
    """

    def __init__(
        self,
        image_keys: tuple = ("cPatch",),
        mask_key:   str   = "pCellmask",
    ):
        self.image_keys = tuple(image_keys)
        self.mask_key   = mask_key

    def __call__(self, sample: dict) -> dict:
        fg = sample[self.mask_key] > 0      # (1, H, W) bool
        for key in self.image_keys:
            if key in sample:
                img = sample[key]           # (C, H, W)
                img[~fg.expand_as(img)] = 0.0
                sample[key] = img
        return sample


# ── Precomputed-stats normalisation ───────────────────────────────────────────

class NormalizeFromStats:
    """Apply precomputed per-channel p1/p99 normalization read from the sample dict.

    Reads four floats from sample[stats_key]:
        mem_lo / mem_hi  — membrane percentiles (computed over pCellmask)
        nuc_lo / nuc_hi  — nuclei   percentiles (computed over pNucmask)

    Membrane channel: normalized over pCellmask pixels.
    Nuclei   channel: normalized over its own non-zero footprint (cnPatches nuclei
                      is exactly 0 outside pNucmask, so img[1] > 0 recovers that mask).
    Background stays 0.  No clamping — matches NormalizeMasked behaviour.
    """

    def __init__(
        self,
        patch_key:  str = "cPatch",
        mask_key:   str = "pCellmask",
        stats_key:  str = "norm_stats",
        eps:        float = 1e-6,
    ):
        self.patch_key = patch_key
        self.mask_key  = mask_key
        self.stats_key = stats_key
        self.eps       = eps

    def __call__(self, sample: dict) -> dict:
        img = sample[self.patch_key]          # (C, H, W) float32
        s   = sample[self.stats_key]
        out = img.clone()

        m = sample[self.mask_key][0] > 0      # pCellmask foreground
        out[0][m] = (img[0][m] - s["mem_lo"]) / (s["mem_hi"] - s["mem_lo"] + self.eps)
        out[0][m] = out[0][m].clamp(0.0, None)  # background
        n = img[1] > 0                         # nuclear footprint (0 outside pNucmask)
        out[1][n] = (img[1][n] - s["nuc_lo"]) / (s["nuc_hi"] - s["nuc_lo"] + self.eps)
        out[1][n] = out[1][n].clamp(0.0, None)  # background

        sample[self.patch_key] = out

        return sample  # optional clamp to [0, 1] after normalisation


# ── Spatial resize (applied last, after all augmentation) ─────────────────────

class ResizePatch:
    """Resize all patch channels to target_size × target_size.

    Uses torchvision v2 type dispatch: TVImage → bilinear (antialias),
    TVMask → nearest neighbour. Applied AFTER normalisation and augmentation
    so that stats and geometric transforms operate at native 256×256 resolution.

    Parameters
    ----------
    target_size   output spatial size (e.g. 96)
    image_keys    sample keys to resize with bilinear interpolation
    mask_keys     sample keys to resize with nearest-neighbour interpolation
    """

    def __init__(
        self,
        target_size: int,
        image_keys: tuple = ("cPatch",),
        mask_keys:  tuple = ("pCellmask", "pNucmask"),
    ):
        self._wrapper = _TVWrapper(
            T.Resize(target_size, antialias=True),
            image_keys=image_keys,
            mask_keys=mask_keys,
        )

    def __call__(self, sample: dict) -> dict:
        return self._wrapper(sample)


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_train_transforms(
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("pCellmask", "pNucmask"),
    norm_mask:  str   = "pCellmask",
    img_size:   int   = 256,
) -> Compose:
    """Training pipeline: normalise → rotate → flip H → flip V → clean background [→ resize].

    Normalization uses precomputed stats from sample["norm_stats"] (written by
    add_cnPatches.py into cell_table.csv and injected by CellPatchDataset.__getitem__).
    Resize (if img_size != 256) is applied last so augmentation runs at full resolution.

    To add a photometric augmentation targeting only images (e.g. Gaussian blur):
        t = build_train_transforms(...)
        t.transforms.insert(-1, _TVWrapper(T.GaussianBlur(5), image_keys=image_keys))
    """
    steps = [
        NormalizeFromStats(mask_key=norm_mask),
        RandomRotate360(image_keys, mask_keys),
        RandomHFlip(image_keys, mask_keys),
        RandomVFlip(image_keys, mask_keys),
        MaskBackground(image_keys),
    ]
    # [256]: no resize step
    if img_size != 256:
        steps.append(ResizePatch(img_size, image_keys=image_keys, mask_keys=("pCellmask", "pNucmask")))
    return Compose(steps)


def build_val_transforms(
    norm_mask: str = "pCellmask",
    img_size:  int = 256,
) -> Compose:
    """Validation / inference pipeline: normalisation only [+ resize], no augmentation."""
    steps = [NormalizeFromStats(mask_key=norm_mask)]
    # [256]: no resize step
    if img_size != 256:
        steps.append(ResizePatch(img_size))
    return Compose(steps)
