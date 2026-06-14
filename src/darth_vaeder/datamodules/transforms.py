"""Per-sample transforms for the multinucleation VAE.

Sample dict (from CellPatchDataset.__getitem__)
-----------------------------------------------
    cPatch      (C, H, W)  float32   isolated masked cell — raw uint16-range
    bbPatch     (C, H, W)  float32   context window       — raw uint16-range (optional)
    cCellmask   (1, H, W)  int64     binary cell mask     — never augmented photometrically
    index       int
    metadata    dict

Transform pipeline (build_transforms)
--------------------------------------
    1. GeometricAug    → same random transform on image_keys AND mask_keys
                         (bilinear for images, nearest for masks — via kornia)
    2. CellPatchNormalize → cPatch to [0,1] using IN-MASK pixels only
       ContextPatchNormalize → bbPatch to [0,1] using all patch pixels (optional)
    3. PhotometricAug  → gamma / blur / noise on image_keys only
                         cCellmask is always skipped here

build_transforms() controls which keys get which transforms via image_keys / mask_keys,
so the pipeline can be reconfigured for different input combinations (e.g. first run
uses only cPatch + cCellmask; later runs add bbPatch).

Geometry before normalisation: raw-value interpolation avoids boundary artefacts
from bilinear resampling on pre-normalised [0,1] values spilling to <0.
Photometric after normalisation: gamma, blur, noise all assume [0,1] input.

All randomness uses the torch RNG (DataLoader reseeds torch RNG per worker/epoch
but does NOT reseed numpy — never draw from numpy here).
"""

from __future__ import annotations

import torch
import kornia.augmentation as K


class Compose:
    """Chain transforms that each map a sample dict to a sample dict."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ── Normalisation (custom — no library does in-mask quantile) ────────────────

class CellPatchNormalize:
    """Scale cPatch to [0, 1] using percentiles of IN-MASK pixels.

    Restricts the intensity statistics to the target cell's interior (cCellmask > 0).
    This anchors the scale to this specific cell and removes cross-replicate exposure
    differences.  Background (zeros) stay at 0 after clamping.

    Parameters
    ----------
    low, high   percentile bounds (default 1 / 99 — robust to hot pixels)
    eps         guard against flat channels (e.g. dead detectors)
    """

    def __init__(self, low: float = 1.0, high: float = 99.0, eps: float = 1e-6):
        self.low  = low / 100.0
        self.high = high / 100.0
        self.eps  = eps

    def __call__(self, sample: dict) -> dict:
        img  = sample["cPatch"]                           # (C, H, W)
        mask = sample["cCellmask"][0].bool().reshape(-1)  # (H*W,)
        qs   = torch.tensor([self.low, self.high], dtype=img.dtype)
        out  = torch.empty_like(img)
        for c in range(img.shape[0]):
            flat = img[c].reshape(-1)
            vals = flat[mask] if mask.any() else flat     # fallback: whole patch
            lo, hi = torch.quantile(vals, qs)
            out[c] = ((img[c] - lo) / (hi - lo + self.eps)).clamp_(0.0, 1.0)
        sample["cPatch"] = out
        return sample


class ContextPatchNormalize:
    """Scale bbPatch to [0, 1] using percentiles of ALL pixels.

    Uses the whole 256×256 context window (including neighbours and background)
    as the reference population, so the brightness of the target cell relative
    to its surroundings is preserved.  Handles cross-replicate gain differences
    because the dominant variance is replicate-wide and affects the whole patch.
    """

    def __init__(self, low: float = 1.0, high: float = 99.0, eps: float = 1e-6):
        self.low  = low / 100.0
        self.high = high / 100.0
        self.eps  = eps

    def __call__(self, sample: dict) -> dict:
        if "bbPatch" not in sample:
            return sample
        img = sample["bbPatch"]                           # (C, H, W)
        qs  = torch.tensor([self.low, self.high], dtype=img.dtype)
        out = torch.empty_like(img)
        for c in range(img.shape[0]):
            lo, hi = torch.quantile(img[c].reshape(-1), qs)
            out[c] = ((img[c] - lo) / (hi - lo + self.eps)).clamp_(0.0, 1.0)
        sample["bbPatch"] = out
        return sample


# ── Geometry augmentations (kornia) ─────────────────────────────────────────

class GeometricAug:
    """Apply the same random geometric transform to images AND masks.

    Uses kornia.augmentation.AugmentationSequential with data_keys to
    guarantee that images (bilinear interpolation) and masks (nearest-neighbour)
    receive the exact same spatial transformation.

    Parameters
    ----------
    image_keys  sample-dict keys to warp as float images (bilinear)
    mask_keys   sample-dict keys to warp as integer masks (nearest)
    p           probability that each sub-augmentation fires
    degrees     max rotation angle in degrees
    """

    def __init__(
        self,
        image_keys: tuple = ("cPatch",),
        mask_keys:  tuple = ("cCellmask",),
        p: float = 0.5,
        degrees: float = 15.0,
    ):
        self.image_keys = tuple(image_keys)
        self.mask_keys  = tuple(mask_keys)
        n_img = len(self.image_keys)
        n_msk = len(self.mask_keys)
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=p),
            K.RandomVerticalFlip(p=p),
            K.RandomRotation(degrees=degrees, p=p),
            data_keys=["input"] * n_img + ["mask"] * n_msk,
            same_on_batch=False,
        )

    def __call__(self, sample: dict) -> dict:
        inputs  = [sample[k].unsqueeze(0) for k in self.image_keys]
        inputs += [sample[k].unsqueeze(0).float() for k in self.mask_keys]
        outs = self.aug(*inputs)
        if isinstance(outs, torch.Tensor):   # single-input → returns tensor
            outs = [outs]
        all_keys = list(self.image_keys) + list(self.mask_keys)
        for key, out in zip(all_keys, outs):
            t = out.squeeze(0)
            sample[key] = t.long() if key in self.mask_keys else t
        return sample


# ── Photometric augmentations (kornia) ──────────────────────────────────────

class PhotometricAug:
    """Apply photometric augmentations to image tensors only.

    Operates on the keys listed in image_keys (default: cPatch).
    cCellmask and any other mask keys are always skipped.

    Same random parameters are used for all image_keys so two views of the same
    FOV (e.g. cPatch + bbPatch) receive consistent photometric perturbation.

    Applied after normalisation so all transforms assume [0, 1] input.
    """

    def __init__(
        self,
        image_keys: tuple = ("cPatch",),
        blur_sigma: tuple = (0.1, 1.2),
        gamma_range: tuple = (0.7, 1.5),
        noise_std: float = 0.05,
        blur_p: float = 0.3,
        gamma_p: float = 0.3,
        noise_p: float = 0.5,
    ):
        self.image_keys = tuple(image_keys)
        n = len(self.image_keys)
        self.aug = K.AugmentationSequential(
            K.RandomGaussianBlur((5, 5), blur_sigma,  p=blur_p),
            K.RandomGamma(gamma_range,                p=gamma_p),
            K.RandomGaussianNoise(mean=0., std=noise_std, p=noise_p),
            data_keys=["input"] * n,
            same_on_batch=False,
        )

    def __call__(self, sample: dict) -> dict:
        inputs = [sample[k].unsqueeze(0) for k in self.image_keys]
        outs   = self.aug(*inputs)
        if isinstance(outs, torch.Tensor):
            outs = [outs]
        for key, out in zip(self.image_keys, outs):
            sample[key] = out.squeeze(0).clamp(0.0, 1.0)
        return sample


# ── Pipeline builder ─────────────────────────────────────────────────────────

def build_transforms(
    train: bool = True,
    image_keys: tuple = ("cPatch",),
    mask_keys:  tuple = ("cCellmask",),
    cell_norm_low:  float = 1.0,
    cell_norm_high: float = 99.0,
    ctx_norm_low:   float = 1.0,
    ctx_norm_high:  float = 99.0,
) -> Compose:
    """Build the per-split transform pipeline.

    Parameters
    ----------
    train           True  → geometry + normalisation + photometric augmentation
                    False → normalisation only (val / test / inference)
    image_keys      which sample-dict keys are float images
                    (geometric + photometric augmentation applied to these)
    mask_keys       which sample-dict keys are integer masks
                    (geometric augmentation only; no photometric)
    cell_norm_low/high  percentile bounds for CellPatchNormalize
    ctx_norm_low/high   percentile bounds for ContextPatchNormalize (bbPatch)

    First-run example (cPatch + cCellmask only)
    -------------------------------------------
        build_transforms(train=True,
                         image_keys=("cPatch",),
                         mask_keys=("cCellmask",))

    Full-model example (cPatch + bbPatch + cCellmask)
    -------------------------------------------------
        build_transforms(train=True,
                         image_keys=("cPatch", "bbPatch"),
                         mask_keys=("cCellmask",))
    """
    # Normalisation is always applied (train and val)
    norm_steps = [CellPatchNormalize(cell_norm_low, cell_norm_high)]
    if "bbPatch" in image_keys:
        norm_steps.append(ContextPatchNormalize(ctx_norm_low, ctx_norm_high))
    normalise = Compose(norm_steps)

    if not train:
        return normalise

    return Compose([
        # 1. geometry: raw values, same spatial transform for images + masks
        GeometricAug(image_keys=image_keys, mask_keys=mask_keys),
        # 2. normalise each patch with its own strategy
        normalise,
        # 3. photometric: [0,1] images only — masks always excluded
        PhotometricAug(image_keys=image_keys),
    ])
