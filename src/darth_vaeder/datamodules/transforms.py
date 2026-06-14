"""On-the-fly transforms for single-cell VAE patches.

Design
------
* ``MaskedPerChannelNormalize`` scales each channel to ``[0, 1]`` using ONLY the
  pixels inside the target-cell mask, so every cell is normalised relative to its
  own intensity — removing field-of-view / exposure bias before the encoder. This
  also matches the sigmoid output of the decoder.
* Geometric augmentations act on the image AND every mask so they stay registered.
* Photometric augmentations act on the image only, after normalisation, and
  re-clamp to ``[0, 1]``.
* All randomness uses the torch RNG. PyTorch's DataLoader reseeds the torch RNG
  per worker per epoch, but does NOT reseed numpy's global RNG — drawing from
  numpy here would make every worker emit identical augmentations.

Every transform takes and returns the sample ``dict`` (keys: ``image``, optional
masks, and the private ``_normmask`` used by the normaliser). Build a pipeline
with :func:`build_transforms`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class Compose:
    """Chain transforms that each map a sample dict to a sample dict."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class MaskedPerChannelNormalize:
    """Scale each channel to ``[0, 1]`` using percentiles of the in-mask pixels.

    Reads ``sample[mask_key]`` (the target cell, shape ``(1, H, W)``). For each
    channel it computes the ``low``/``high`` percentiles over the cell-interior
    pixels, rescales the whole patch, and clamps to ``[0, 1]``. Background (and
    bbPatch neighbours) fall outside ``[lo, hi]`` and clamp to 0/1.
    """

    def __init__(self, low: float = 1.0, high: float = 99.0,
                 mask_key: str = "_normmask", eps: float = 1e-6):
        self.low, self.high, self.mask_key, self.eps = low, high, mask_key, eps

    def __call__(self, sample: dict) -> dict:
        img = sample["image"]
        m = sample[self.mask_key]
        if m.ndim == 3:
            m = m[0]
        m = m.bool().reshape(-1)
        out = torch.empty_like(img)
        qs = torch.tensor([self.low / 100.0, self.high / 100.0], dtype=img.dtype)
        for c in range(img.shape[0]):
            flat = img[c].reshape(-1)
            vals = flat[m] if m.any() else flat            # fallback: whole patch
            lo, hi = torch.quantile(vals, qs)
            out[c] = ((img[c] - lo) / (hi - lo + self.eps)).clamp_(0.0, 1.0)
        sample["image"] = out
        return sample


class RandomFlipRotate90:
    """Exact flips + 90-degree rotations (no interpolation). Acts on image+masks."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        do_h = torch.rand(()) < self.p
        do_v = torch.rand(()) < self.p
        k = int(torch.randint(0, 4, (1,)))
        for key, v in sample.items():
            if torch.is_tensor(v) and v.ndim == 3:
                t = v
                if do_h:
                    t = torch.flip(t, [-1])
                if do_v:
                    t = torch.flip(t, [-2])
                if k:
                    t = torch.rot90(t, k, [-2, -1])
                sample[key] = t.contiguous()
        return sample


class RandomAffine:
    """Random rotation / scale / translation. Bilinear image, nearest masks."""

    def __init__(self, degrees: float = 15.0, scale: tuple[float, float] = (0.9, 1.1),
                 translate: float = 0.05, p: float = 0.5):
        self.degrees, self.scale, self.translate, self.p = degrees, scale, translate, p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        ang = (torch.rand(()) * 2 - 1) * self.degrees * math.pi / 180
        sc = self.scale[0] + torch.rand(()) * (self.scale[1] - self.scale[0])
        inv = 1.0 / sc
        tx = (torch.rand(()) * 2 - 1) * self.translate * 2
        ty = (torch.rand(()) * 2 - 1) * self.translate * 2
        cs, sn = torch.cos(ang), torch.sin(ang)
        theta = torch.tensor([[cs * inv, -sn * inv, tx],
                              [sn * inv, cs * inv, ty]], dtype=torch.float32)[None]
        for key, v in list(sample.items()):
            if torch.is_tensor(v) and v.ndim == 3:
                c, h, w = v.shape
                grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
                mode = "bilinear" if key == "image" else "nearest"
                out = F.grid_sample(v[None].float(), grid, mode=mode,
                                    padding_mode="zeros", align_corners=False)[0]
                sample[key] = out if key == "image" else out.to(v.dtype)
        return sample


class RandomGamma:
    """Non-linear contrast jitter on the normalised ``[0, 1]`` image."""

    def __init__(self, gamma: tuple[float, float] = (0.7, 1.5), p: float = 0.3):
        self.gamma, self.p = gamma, p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) < self.p:
            g = self.gamma[0] + torch.rand(()) * (self.gamma[1] - self.gamma[0])
            sample["image"] = sample["image"].clamp(0, 1).pow(g)
        return sample


class RandomGaussianBlur:
    """Mild separable Gaussian blur (simulates focus variation)."""

    def __init__(self, sigma: tuple[float, float] = (0.1, 1.2), kernel: int = 5, p: float = 0.3):
        self.sigma, self.kernel, self.p = sigma, kernel, p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        sig = self.sigma[0] + torch.rand(()) * (self.sigma[1] - self.sigma[0])
        k = self.kernel
        ax = torch.arange(k) - k // 2
        g = torch.exp(-(ax**2) / (2 * sig * sig))
        g = (g / g.sum()).to(torch.float32)
        img = sample["image"]
        c = img.shape[0]
        kx = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
        ky = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
        x = F.conv2d(img[None], kx, padding=(0, k // 2), groups=c)
        x = F.conv2d(x, ky, padding=(k // 2, 0), groups=c)
        sample["image"] = x[0].clamp_(0, 1)
        return sample


class RandomGaussianNoise:
    """Additive sensor-style noise on the normalised image (apply last)."""

    def __init__(self, std: float = 0.05, p: float = 0.5):
        self.std, self.p = std, p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()) < self.p:
            sample["image"] = (sample["image"] + torch.randn_like(sample["image"]) * self.std).clamp_(0, 1)
        return sample


def build_transforms(train: bool, norm_low: float = 1.0, norm_high: float = 99.0) -> Compose:
    """Build the per-split pipeline.

    Validation/test = masked normalisation only. Training =
    geometry -> masked normalisation -> photometric augmentations.
    """
    norm = MaskedPerChannelNormalize(low=norm_low, high=norm_high)
    if not train:
        return Compose([norm])
    return Compose([
        RandomFlipRotate90(p=0.5),
        RandomAffine(degrees=15, scale=(0.9, 1.1), translate=0.05, p=0.5),
        norm,                                       # normalise AFTER geometry
        RandomGamma(gamma=(0.7, 1.5), p=0.3),
        RandomGaussianBlur(sigma=(0.1, 1.2), p=0.3),
        RandomGaussianNoise(std=0.05, p=0.5),       # noise last
    ])
