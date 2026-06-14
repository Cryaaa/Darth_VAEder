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
        mask_key:  str = "cCellmask",
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
        mask_keys:  tuple = ("cCellmask",),
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
    mask_keys:  tuple = ("cCellmask",),
) -> _TVWrapper:
    """Uniform random rotation in [0°, 360°) — bilinear for images, nearest for masks."""
    return _TVWrapper(
        T.RandomRotation(degrees=(0, 360), fill=0),
        image_keys, mask_keys,
    )


def RandomHFlip(
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("cCellmask",),
    p: float = 0.5,
) -> _TVWrapper:
    """Random horizontal flip with probability p."""
    return _TVWrapper(T.RandomHorizontalFlip(p=p), image_keys, mask_keys)


def RandomVFlip(
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("cCellmask",),
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
        mask_key:   str   = "cCellmask",
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


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_train_transforms(
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("cCellmask",),
    norm_low:   float = 1.0,
    norm_high:  float = 99.0,
) -> Compose:
    """Training pipeline: normalise → rotate → flip H → flip V → clean background.

    To add a photometric augmentation targeting only images (e.g. Gaussian blur):
        t = build_train_transforms(...)
        t.transforms.insert(-1, _TVWrapper(T.GaussianBlur(5), image_keys=image_keys))
    Or replace dm.train_transform with a fully custom Compose.
    """
    return Compose([
        NormalizeMasked(low=norm_low, high=norm_high),
        RandomRotate360(image_keys, mask_keys),
        RandomHFlip(image_keys, mask_keys),
        RandomVFlip(image_keys, mask_keys),
        MaskBackground(image_keys),
    ])


def build_val_transforms(
    norm_low:  float = 1.0,
    norm_high: float = 99.0,
) -> Compose:
    """Validation / inference pipeline: normalisation only, no augmentation."""
    return Compose([NormalizeMasked(low=norm_low, high=norm_high)])
