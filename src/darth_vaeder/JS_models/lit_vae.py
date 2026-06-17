"""LightningModule wrapping VAEResNet18 for single-cell representation learning.

Loss = recon_weight * MSE_mean(all 4 ch)
       + beta        * KL_per_dim_mean
       + ssim_weight * (SSIM_membrane + SSIM_nuclei_masked)

All three terms use per-element MEAN reductions so their magnitudes are invariant
to img_size, nc and z_dim — the weights are then directly interpretable relative
importances (changing z_dim no longer silently rescales the effective beta).
Natural per-element magnitudes: recon ~1.5e-3, kl ~0.9 (per dim), ssim ~0.2.

Input layout
------------
    batch["cPatch"]    (B, 2, H, W)  normalised cnPatches, background=0
    batch["pCellmask"] (B, 1, H, W)  dilated crop mask (int64)
    batch["pNucmask"]  (B, 1, H, W)  dilated nuclear mask (int64)

    → encoder input: cat([cPatch, pCellmask.float(), pNucmask.float()], dim=1)  (B, 4, H, W)

    recon channel order (dual decoder output, see vae.py):
        ch0 = membrane  (decoderCell ch0)
        ch1 = nuclei    (decoderNuc  ch0)
        ch2 = pCellmask (decoderCell ch1)
        ch3 = pNucmask  (decoderNuc  ch1)

SSIM
----
    Membrane: windowed SSIM on recon[:,0:1] vs x_in[:,0:1], unmasked mean (11×11 window).
    Nuclei:   windowed SSIM on recon[:,1:2] vs x_in[:,1:2], mean over pNucmask pixels only.
    ssim_weight=0 disables SSIM entirely (backward compatible with old checkpoints).
"""

import torch
import torch.nn.functional as F
import lightning as L

from .vae import VAEResNet18


def _gaussian_window(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1-D Gaussian outer-producted to a 2-D window, shape (1, 1, k, k)."""
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = g.outer(g)
    return w.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)


class LitVAE(L.LightningModule):
    """
    Parameters
    ----------
    nc              total input channels fed to the encoder (4 = 2 image + 2 masks)
    nc_img          number of image-only channels
    image_key       batch dict key for the image patch        (default "cPatch")
    mask_key        batch dict key for the crop mask          (default "pCellmask")
    nuc_mask_key    batch dict key for the nuclear mask       (default "pNucmask")
    z_dim           latent dimensionality                     (default 10)
    beta            KL weight; updated at runtime by BetaAnnealing callback
    lr              Adam learning rate                        (default 1e-3)
    img_size        spatial patch size                        (default 256)
    ssim_weight     weight on SSIM loss; 0 = disabled         (default 0.0)
    recon_weight    weight on per-element-mean MSE recon      (default 300.0)
    """

    def __init__(
        self,
        nc: int = 4,
        nc_img: int = 2,
        image_key: str = "cPatch",
        mask_key: str = "pCellmask",
        nuc_mask_key: str = "pNucmask",
        recon_function=F.mse_loss,
        z_dim: int = 10,
        beta: float = 0,
        lr: float = 1e-3,
        img_size: int = 256,
        ssim_weight: float = 0.0,
        recon_weight: float = 300.0,
    ):
        super().__init__()
        self.nc_img         = nc_img
        self.recon_function = recon_function
        self.save_hyperparameters(ignore=["recon_function"])
        self.vae  = VAEResNet18(nc=nc, z_dim=z_dim, img_size=img_size)
        self.beta = beta
        self.lr   = lr
        # Gaussian window for SSIM — persistent=False so it is NOT saved in checkpoints.
        # This means warm-starting from old checkpoints (which lack this buffer) works fine.
        self.register_buffer("_ssim_win", _gaussian_window(11, 1.5), persistent=False)

    # ── SSIM helpers ──────────────────────────────────────────────────────────

    def _ssim_map(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Per-pixel SSIM map.

        pred, tgt : (B, 1, H, W)
        returns   : (B, 1, H, W)  values nominally in [0, 1]

        Uses same-padding (pad = kernel//2) so the output is the same spatial
        size as the input — necessary for element-wise mask multiplication.
        """
        w   = self._ssim_win        # (1, 1, 11, 11) — on same device as inputs
        pad = w.shape[-1] // 2     # 5
        C1, C2 = 0.01 ** 2, 0.03 ** 2   # stability constants (data_range=1)

        mu1    = F.conv2d(pred,       w, padding=pad)
        mu2    = F.conv2d(tgt,        w, padding=pad)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12   = mu1 * mu2

        sig1_sq = F.conv2d(pred * pred, w, padding=pad) - mu1_sq
        sig2_sq = F.conv2d(tgt  * tgt,  w, padding=pad) - mu2_sq
        sig12   = F.conv2d(pred * tgt,  w, padding=pad) - mu12

        num = (2 * mu12            + C1) * (2 * sig12            + C2)
        den = (mu1_sq + mu2_sq     + C1) * (sig1_sq + sig2_sq    + C2)
        return num / den   # (B, 1, H, W)

    # ── Core step ─────────────────────────────────────────────────────────────

    def forward(self, x):
        """x: (B, nc, H, W) → recon, z, mu, logvar"""
        return self.vae(x)

    def _step(self, batch):
        x_img    = batch[self.hparams.image_key]       # (B, 2, H, W) — cnPatches
        mask     = batch[self.hparams.mask_key]        # (B, 1, H, W) — pCellmask
        nuc_mask = batch[self.hparams.nuc_mask_key]    # (B, 1, H, W) — pNucmask

        x_in  = torch.cat([x_img, mask.float(), nuc_mask.float()], dim=1)  # (B, 4, H, W)
        recon, z, mu, logvar = self.vae(x_in)

        # MSE — per-element mean (invariant to img_size & nc); weighted by recon_weight
        recon_loss = self.recon_function(recon, x_in, reduction="mean")

        # KL divergence — per-dim mean (invariant to z_dim): sum over dims, mean over
        # batch, then divide by z_dim so the magnitude doesn't scale with latent size
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
        ) / mu.shape[1]

        # SSIM structural loss — only computed when ssim_weight > 0
        if self.hparams.ssim_weight > 0:
            # membrane: unmasked SSIM over the full patch
            ssim_mem = self._ssim_map(recon[:, 0:1], x_in[:, 0:1]).mean()

            # nuclei: SSIM averaged only over pixels inside pNucmask
            nuc_m        = (nuc_mask > 0).float()                          # (B, 1, H, W)
            ssim_map_nuc = self._ssim_map(recon[:, 1:2], x_in[:, 1:2])    # (B, 1, H, W)
            ssim_nuc     = (ssim_map_nuc * nuc_m).sum() / (nuc_m.sum() + 1e-6)

            ssim_loss = (1 - ssim_mem) + (1 - ssim_nuc)
        else:
            ssim_mem  = torch.zeros(1, device=self.device)
            ssim_nuc  = torch.zeros(1, device=self.device)
            ssim_loss = torch.zeros(1, device=self.device)

        loss = (
            self.hparams.recon_weight * recon_loss
            + self.beta               * kl_loss
            + self.hparams.ssim_weight * ssim_loss
        )
        return loss, recon_loss, kl_loss, ssim_mem, ssim_nuc

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss, recon, kl, ssim_mem, ssim_nuc = self._step(batch)
        self.log_dict(
            {
                "train/loss": loss, "train/recon": recon, "train/kl": kl,
                "train/ssim_mem": ssim_mem, "train/ssim_nuc": ssim_nuc,
            },
            on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, kl, ssim_mem, ssim_nuc = self._step(batch)
        self.log_dict(
            {
                "val/loss": loss, "val/recon": recon, "val/kl": kl,
                "val/ssim_mem": ssim_mem, "val/ssim_nuc": ssim_nuc,
            },
            on_step=False, on_epoch=True, prog_bar=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
