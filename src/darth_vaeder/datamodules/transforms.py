"""Per-sample transforms for the dual-input single-cell VAE.

Sample dict keys
----------------
    cPatch      (C, H, W)  float32   isolated masked cell — raw uint16-range values
    bbPatch     (C, H, W)  float32   context window       — raw uint16-range values
    cCellmask   (1, H, W)  int64     binary cell mask     — NOT normalised, NOT augmented
    index       int
    metadata    dict  (added after transform)

Transform order (training)
--------------------------
    1. Geometry      RandomFlipRotate90, RandomAffine
                     → same spatial transform applied identically to cPatch, bbPatch,
                       and cCellmask (bilinear for images, nearest for mask)
    2. Normalise     CellPatchNormalize  → cPatch  to [0,1] using IN-MASK pixels
                     ContextPatchNormalize → bbPatch to [0,1] using ALL patch pixels
                     cCellmask is left unchanged.
    3. Photometric   RandomGamma, RandomGaussianBlur, RandomGaussianNoise
                     → applied to cPatch AND bbPatch with the SAME random parameter
                       (both come from the same acquisition / light condition)
                     → cCellmask is skipped.

Validation / inference: steps 2 only (no geometry or photometric noise).

All randomness uses the torch RNG.  PyTorch DataLoader reseeds torch RNG per
worker per epoch but does NOT reseed numpy — never draw from numpy here or
workers will produce identical augmentations.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Keys treated as floating-point images (bilinear warp, photometric augmentation)
_IMG_KEYS  = ("cPatch", "bbPatch")
# Keys treated as integer label masks (nearest warp, no photometric augmentation)
_MASK_KEYS = ("cCellmask",)


class Compose:
    """Chain transforms that each map a sample dict to a sample dict."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ── Normalisation ────────────────────────────────────────────────────────────

class CellPatchNormalize:
    """Scale cPatch channels to [0, 1] using quantiles of the IN-MASK pixels.

    Reads cCellmask to restrict statistics to the target cell's interior.
    Background pixels and any neighbouring cells that bleed into the crop are
    excluded from the percentile calculation, so the normalisation is anchored
    to THIS cell's intensity range.  After scaling the whole patch is clamped,
    meaning background (zeros) stay at 0 and any bright outliers clip at 1.

    Parameters
    ----------
    low, high   percentile bounds (default 1 / 99 — robust to hot pixels)
    eps         numerical guard against flat cells (e.g. dead channels)
    """

    def __init__(self, low: float = 1.0, high: float = 99.0, eps: float = 1e-6):
        self.low  = low / 100.0
        self.high = high / 100.0
        self.eps  = eps

    def __call__(self, sample: dict) -> dict:
        img  = sample["cPatch"]                         # (C, H, W)
        mask = sample["cCellmask"][0].bool().reshape(-1)  # (H*W,)
        qs   = torch.tensor([self.low, self.high], dtype=img.dtype)
        out  = torch.empty_like(img)
        for c in range(img.shape[0]):
            flat = img[c].reshape(-1)
            vals = flat[mask] if mask.any() else flat   # fallback: whole patch
            lo, hi = torch.quantile(vals, qs)
            out[c] = ((img[c] - lo) / (hi - lo + self.eps)).clamp_(0.0, 1.0)
        sample["cPatch"] = out
        return sample


class ContextPatchNormalize:
    """Scale bbPatch channels to [0, 1] using quantiles of ALL patch pixels.

    Unlike CellPatchNormalize this uses the entire 256x256 window — including
    neighbouring cells and background — as the reference population.  This
    preserves the relative brightness of the target cell against its context,
    which is the information the encoder is supposed to use from bbPatch.

    Note: because bbPatch is a crop of one FOV, its pixel statistics approximate
    the local image statistics but are not identical to whole-slide statistics.
    For the purpose of removing cross-replicate acquisition bias this is
    sufficient (the dominant variance in raw intensity is replicate-level gain,
    which affects the whole patch uniformly).
    """

    def __init__(self, low: float = 1.0, high: float = 99.0, eps: float = 1e-6):
        self.low  = low / 100.0
        self.high = high / 100.0
        self.eps  = eps

    def __call__(self, sample: dict) -> dict:
        img = sample["bbPatch"]                         # (C, H, W)
        qs  = torch.tensor([self.low, self.high], dtype=img.dtype)
        out = torch.empty_like(img)
        for c in range(img.shape[0]):
            lo, hi = torch.quantile(img[c].reshape(-1), qs)
            out[c] = ((img[c] - lo) / (hi - lo + self.eps)).clamp_(0.0, 1.0)
        sample["bbPatch"] = out
        return sample


# ── Geometry augmentations ───────────────────────────────────────────────────

class RandomFlipRotate90:
    """Exact flips + 90° rotations (no interpolation artefacts).

    The same flip/rotation is applied to cPatch, bbPatch, and cCellmask so
    all three stay spatially registered with each other.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        do_h = torch.rand(()) < self.p
        do_v = torch.rand(()) < self.p
        k    = int(torch.randint(0, 4, (1,)))
        for key in _IMG_KEYS + _MASK_KEYS:
            v = sample.get(key)
            if v is None or not torch.is_tensor(v):
                continue
            if do_h: v = torch.flip(v, [-1])
            if do_v: v = torch.flip(v, [-2])
            if k:    v = torch.rot90(v, k, [-2, -1])
            sample[key] = v.contiguous()
        return sample


class RandomAffine:
    """Random rotation / isotropic scale / translation.

    A single affine matrix is sampled and applied to all spatial tensors:
    bilinear for cPatch and bbPatch (continuous intensities), nearest-neighbour
    for cCellmask (integer labels — no interpolation artefacts).
    """

    def __init__(self, degrees: float = 15.0,
                 scale: tuple[float, float] = (0.9, 1.1),
                 translate: float = 0.05,
                 p: float = 0.5):
        self.degrees   = degrees
        self.scale     = scale
        self.translate = translate
        self.p         = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample

        ang = (torch.rand(()) * 2 - 1) * self.degrees * math.pi / 180
        sc  = self.scale[0] + torch.rand(()) * (self.scale[1] - self.scale[0])
        inv = 1.0 / sc
        tx  = (torch.rand(()) * 2 - 1) * self.translate * 2
        ty  = (torch.rand(()) * 2 - 1) * self.translate * 2
        cs, sn = torch.cos(ang), torch.sin(ang)
        theta  = torch.tensor(
            [[cs * inv, -sn * inv, tx],
             [sn * inv,  cs * inv, ty]], dtype=torch.float32
        ).unsqueeze(0)                                  # (1, 2, 3)

        for key in _IMG_KEYS + _MASK_KEYS:
            v = sample.get(key)
            if v is None or not torch.is_tensor(v):
                continue
            c, h, w = v.shape
            grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
            mode = "nearest" if key in _MASK_KEYS else "bilinear"
            out  = F.grid_sample(
                v.unsqueeze(0).float(), grid,
                mode=mode, padding_mode="zeros", align_corners=False
            )[0]
            sample[key] = out if key in _IMG_KEYS else out.to(v.dtype)
        return sample


# ── Photometric augmentations ────────────────────────────────────────────────
# These operate on normalised [0, 1] images only.
# The SAME random parameter is used for cPatch and bbPatch — both were acquired
# under the same microscope settings for this FOV, so the same perturbation is
# physically consistent.  cCellmask is always skipped.

class RandomGamma:
    """Contrast jitter via power-law transform on [0, 1] images.

    gamma < 1 brightens, gamma > 1 darkens.  Output stays in [0, 1].
    """

    def __init__(self, gamma: tuple[float, float] = (0.7, 1.5), p: float = 0.3):
        self.gamma = gamma
        self.p     = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        g = self.gamma[0] + torch.rand(()) * (self.gamma[1] - self.gamma[0])
        for key in _IMG_KEYS:
            if key in sample:
                sample[key] = sample[key].clamp(0, 1).pow(g)
        return sample


class RandomGaussianBlur:
    """Separable Gaussian blur (simulates mild focus variation).

    Applied to both image patches with the same kernel so the spatial
    frequency content of cPatch and bbPatch degrades consistently.
    """

    def __init__(self, sigma: tuple[float, float] = (0.1, 1.2),
                 kernel: int = 5, p: float = 0.3):
        self.sigma  = sigma
        self.kernel = kernel
        self.p      = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        sig = self.sigma[0] + torch.rand(()) * (self.sigma[1] - self.sigma[0])
        k   = self.kernel
        ax  = torch.arange(k) - k // 2
        g   = torch.exp(-(ax ** 2) / (2 * sig * sig))
        g   = (g / g.sum()).float()
        for key in _IMG_KEYS:
            img = sample.get(key)
            if img is None:
                continue
            c  = img.shape[0]
            kx = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
            ky = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
            x  = F.conv2d(img.unsqueeze(0), kx, padding=(0, k // 2), groups=c)
            x  = F.conv2d(x, ky, padding=(k // 2, 0), groups=c)
            sample[key] = x[0].clamp_(0, 1)
        return sample


class RandomGaussianNoise:
    """Additive Gaussian noise (models camera/detector shot noise).

    Apply last — after blur — so the noise is not spatially correlated.
    Same noise level for both patches; independent noise realisations.
    """

    def __init__(self, std: float = 0.05, p: float = 0.5):
        self.std = std
        self.p   = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        for key in _IMG_KEYS:
            if key in sample:
                sample[key] = (
                    sample[key] + torch.randn_like(sample[key]) * self.std
                ).clamp_(0, 1)
        return sample


# ── Pipeline builder ─────────────────────────────────────────────────────────

def build_transforms(
    train: bool = True,
    cell_norm_low:  float = 1.0,
    cell_norm_high: float = 99.0,
    ctx_norm_low:   float = 1.0,
    ctx_norm_high:  float = 99.0,
) -> Compose:
    """Build the standard per-split transform pipeline.

    Parameters
    ----------
    train           True → geometry + normalisation + photometric augmentations.
                    False → normalisation only (val / test / inference).
    cell_norm_low/high  percentile bounds for CellPatchNormalize (cPatch)
    ctx_norm_low/high   percentile bounds for ContextPatchNormalize (bbPatch)
    """
    normalise = Compose([
        CellPatchNormalize(cell_norm_low, cell_norm_high),
        ContextPatchNormalize(ctx_norm_low, ctx_norm_high),
    ])
    if not train:
        return normalise

    return Compose([
        # 1. geometry — before normalisation so intensity stats are not distorted
        #    by interpolation artefacts at the crop boundary
        RandomFlipRotate90(p=0.5),
        RandomAffine(degrees=15, scale=(0.9, 1.1), translate=0.05, p=0.5),
        # 2. normalise each patch with its own strategy
        normalise,
        # 3. photometric — after normalisation so they operate in [0, 1]
        RandomGamma(gamma=(0.7, 1.5), p=0.3),
        RandomGaussianBlur(sigma=(0.1, 1.2), p=0.3),
        RandomGaussianNoise(std=0.05, p=0.5),           # noise always last
    ])
