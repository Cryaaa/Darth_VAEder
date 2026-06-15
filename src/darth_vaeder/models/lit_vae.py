"""LightningModule wrapping VAEResNet18 for single-cell representation learning.

Loss = reconstruction (MSE on all input channels) + beta * KL.

Input layout
------------
    batch["cPatch"]    (B, 2, 256, 256)  normalised cnPatches, background=0
    batch["pCellmask"] (B, 1, 256, 256)  dilated crop mask (int64)
    batch["pNucmask"]  (B, 1, 256, 256)  dilated nuclear mask (int64)

    → encoder input: cat([cPatch, pCellmask.float(), pNucmask.float()], dim=1)  (B, 4, H, W)

Latent
------
    Flat bottleneck: mu, logvar  (B, z_dim)  — global per-cell vector.
"""

import torch
import torch.nn.functional as F
import lightning as L

from .vae import VAEResNet18


class LitVAE(L.LightningModule):
    """
    Parameters
    ----------
    nc              total input channels fed to the encoder
                    (4 = 2 cPatch channels + 1 mask channel)
    nc_img          number of image-only channels; loss computed on recon[:, :nc_img]
    image_key       batch dict key for the image patch   (default "cPatch")
    mask_key        batch dict key for the crop mask     (default "pCellmask")
    recon_function  pixel-level loss function            (default F.mse_loss)
    z_dim           latent dimensionality                (default 10)
    beta            KL weight; 0 = pure reconstruction  (default 1.0)
    lr              Adam learning rate                   (default 1e-3)
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
        img_size: int = 256,   # [256]: no img_size param; set to 96 for downsampled mode
    ):
        super().__init__()
        self.nc_img         = nc_img
        self.recon_function = recon_function
        self.save_hyperparameters(ignore=["recon_function"])
        # [256]: self.vae = VAEResNet18(nc=nc, z_dim=z_dim)
        self.vae  = VAEResNet18(nc=nc, z_dim=z_dim, img_size=img_size)
        self.beta = beta
        self.lr   = lr

    def forward(self, x):
        """x: (B, nc, H, W) → recon, z, mu, logvar"""
        return self.vae(x)

    def _step(self, batch):
        x_img    = batch[self.hparams.image_key]       # (B, 2, H, W) — cnPatches
        mask     = batch[self.hparams.mask_key]        # (B, 1, H, W) — pCellmask
        nuc_mask = batch[self.hparams.nuc_mask_key]    # (B, 1, H, W) — pNucmask

        # concatenate masks as channels 3 and 4
        x_in  = torch.cat([x_img, mask.float(), nuc_mask.float()], dim=1)  # (B, 4, H, W)
        recon, z, mu, logvar = self.vae(x_in)

        # sum over pixels/channels, mean over batch — standard ELBO scaling
        recon_loss = self.recon_function(recon, x_in, reduction="sum") / recon.shape[0]
        # sum over latent dims, mean over batch
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
        loss = recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        loss, recon, kl = self._step(batch)
        self.log_dict(
            {"train/loss": loss, "train/recon": recon, "train/kl": kl},
            on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, kl = self._step(batch)
        self.log_dict(
            {"val/loss": loss, "val/recon": recon, "val/kl": kl},
            on_step=False, on_epoch=True, prog_bar=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
