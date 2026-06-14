"""On-the-fly transforms for the multinucleation VAE.

Design principles
-----------------
* Pure torch — no external augmentation library.
* Every transform takes and returns the sample dict.
* Geometric transforms have explicit image_keys / mask_keys so you can target
  any subset of the sample dict.  Add future augmentations the same way.
* NormalizeMasked has explicit patch_key / mask_key for the same reason.
* Normalisation comes BEFORE geometry: stats are computed on raw values, then
  geometry interpolates the already-normalised patch.

Pipeline (first run — cPatch + cCellmask only)
----------------------------------------------
    Compose([
        NormalizeMasked(),       # percentile on in-mask pixels, bg stays 0, no clamp
        RandomRotate360(),       # uniform [0, 360) — same angle for image + mask
        RandomHFlip(),           # p=0.5 — same flip for image + mask
        RandomVFlip(),           # p=0.5 — same flip for image + mask
    ])

To add a future augmentation that targets only cPatch (e.g. gaussian noise):
    class RandomNoise:
        def __init__(self, image_keys=("cPatch",), std=0.05, p=0.5): ...
and append it to the Compose list.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


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

    Stats (p_low, p_high) are computed from the pixels where mask_key > 0.
    The scale  (x - p_low) / (p_high - p_low)  is then applied ONLY to those
    same in-mask pixels.  Background pixels (mask == 0) are left at exactly 0.

    No clamping — pixels brighter than p_high map to > 1; this is intentional
    so the network sees the full dynamic range of outlier pixels.

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
        img  = sample[self.patch_key]                # (C, H, W)  float32  raw
        mask = sample[self.mask_key][0] > 0          # (H, W)     bool
        out  = img.clone()                           # copy — background stays 0
        qs   = self.qs.to(img.dtype)

        for c in range(img.shape[0]):
            pixels = img[c][mask]                    # 1-D tensor of in-mask values
            if pixels.numel() == 0:
                continue                             # empty mask — leave channel as-is
            lo, hi = torch.quantile(pixels, qs)
            # apply only to in-mask positions; background is untouched (stays 0)
            out[c][mask] = (img[c][mask] - lo) / (hi - lo + self.eps)

        sample[self.patch_key] = out
        return sample


# ── Geometric augmentations ───────────────────────────────────────────────────
# All geometric transforms share the same interface:
#   image_keys  → warped with bilinear interpolation (float tensors)
#   mask_keys   → warped with nearest-neighbour      (integer label tensors)
# The same random parameters are used for all keys so they stay registered.

def _warp(tensor: torch.Tensor, theta: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a 2×3 affine theta to a (C, H, W) tensor."""
    grid = F.affine_grid(theta, (1,) + tensor.shape, align_corners=False)
    out  = F.grid_sample(
        tensor.unsqueeze(0).float(), grid,
        mode=mode, padding_mode="zeros", align_corners=False,
    ).squeeze(0)
    return out


class RandomRotate360:
    """Rotate by a uniformly sampled angle in [0°, 360°).

    Uses F.affine_grid + F.grid_sample:
        bilinear interpolation for image tensors  (image_keys)
        nearest-neighbour for integer mask tensors (mask_keys)
    The SAME angle is applied to every key so cPatch and cCellmask stay aligned.
    """

    def __init__(
        self,
        image_keys: tuple = ("cPatch",),
        mask_keys:  tuple = ("cCellmask",),
    ):
        self.image_keys = tuple(image_keys)
        self.mask_keys  = tuple(mask_keys)

    def __call__(self, sample: dict) -> dict:
        angle = torch.rand(()) * 2 * math.pi          # uniform [0, 2π)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0],
             [sin_a,  cos_a, 0.0]], dtype=torch.float32,
        ).unsqueeze(0)                                 # (1, 2, 3)

        for key in self.image_keys:
            if key in sample:
                sample[key] = _warp(sample[key], theta, mode="bilinear")

        for key in self.mask_keys:
            if key in sample:
                sample[key] = _warp(sample[key], theta, mode="nearest").to(sample[key].dtype)

        return sample


class RandomHFlip:
    """Random horizontal flip (left ↔ right).  Same flip for all keys."""

    def __init__(
        self,
        image_keys: tuple = ("cPatch",),
        mask_keys:  tuple = ("cCellmask",),
        p: float = 0.5,
    ):
        self.image_keys = tuple(image_keys)
        self.mask_keys  = tuple(mask_keys)
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        for key in self.image_keys + self.mask_keys:
            if key in sample:
                sample[key] = torch.flip(sample[key], dims=[-1])
        return sample


class RandomVFlip:
    """Random vertical flip (top ↔ bottom).  Same flip for all keys."""

    def __init__(
        self,
        image_keys: tuple = ("cPatch",),
        mask_keys:  tuple = ("cCellmask",),
        p: float = 0.5,
    ):
        self.image_keys = tuple(image_keys)
        self.mask_keys  = tuple(mask_keys)
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        for key in self.image_keys + self.mask_keys:
            if key in sample:
                sample[key] = torch.flip(sample[key], dims=[-2])
        return sample


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_train_transforms(
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("cCellmask",),
    norm_low:   float = 1.0,
    norm_high:  float = 99.0,
) -> Compose:
    """Training pipeline: normalise → rotate → flip H → flip V.

    To add a new augmentation that targets only images (e.g. noise):
        transforms = build_train_transforms(...)
        transforms.transforms.append(MyAug(image_keys=image_keys))
    Or pass a custom Compose entirely via dm.train_transform = ...
    """
    return Compose([
        NormalizeMasked(low=norm_low, high=norm_high),
        RandomRotate360(image_keys, mask_keys),
        RandomHFlip(image_keys, mask_keys),
        RandomVFlip(image_keys, mask_keys),
        MaskBackground(image_keys),          # re-zero background after bilinear rotation
    ])


def build_val_transforms(
    norm_low:  float = 1.0,
    norm_high: float = 99.0,
) -> Compose:
    """Validation / inference pipeline: normalisation only, no augmentation."""
    return Compose([NormalizeMasked(low=norm_low, high=norm_high)])


class MaskBackground:
    """Zero out image pixels wherever the mask is 0.

    Bilinear rotation smears a thin fringe of cell pixels into the background.
    Applying this after all geometric transforms restores the invariant that
    background == 0, without affecting the in-mask cell content.

    Parameters
    ----------
    image_keys  keys to zero-out in background regions
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
        fg = sample[self.mask_key] > 0          # (1, H, W) bool
        for key in self.image_keys:
            if key in sample:
                img = sample[key]               # (C, H, W)
                img[~fg.expand_as(img)] = 0.0
                sample[key] = img
        return sample
