"""LightningModule wrapping VAEResNet18 for single-cell representation learning.

Loss = masked reconstruction (MSE inside pCellmask) + beta * KL divergence.
pCellmask is the dilated crop mask that matches the true extent of cPatches.
The mask ensures the VAE is scored only on cell pixels, not the zero background.

Input (first run, include_bb=False)
------------------------------------
    batch["cPatch"]    (B, nc, 256, 256)  normalised, background=0
    batch["pCellmask"] (B, 1,  256, 256)  dilated crop mask (int64)

The encoder produces spatial mu / logvar (B, z_dim, 16, 16).
The decoder reconstructs (B, nc, 256, 256) with sigmoid activation.
"""

import torch
import torch.nn.functional as F
import lightning as L

from .vae import VAEResNet18


class LitVAE(L.LightningModule):
    """
    Parameters
    ----------
    nc      number of input/output channels (2: membrane + nuclei)
    z_dim   latent dimensionality per spatial position (encoder output: z_dim × 16 × 16)
    beta    weight on the KL term — beta=1 is standard VAE; >1 is beta-VAE
    lr      Adam learning rate
    """

    def __init__(self,  
                  nc: int = 2,
                  image_key: str = "cPatch",
                  mask_key: str = "pCellmask",
                  recon_function = F.mse_loss,
                  z_dim: int = 64,
                  beta: float = 1.0,
                  lr: float = 1e-3):
        super().__init__()
        self.recon_function = recon_function
        self.save_hyperparameters()
        self.vae  = VAEResNet18(nc=nc, z_dim=z_dim)
        self.image_key = image_key
        self.mask_key = mask_key
        self.beta = beta
        self.lr   = lr

    def forward(self, x):
        """x: (B, nc, H, W) → recon, z, mu, logvar"""
        return self.vae(x)

    def _step(self, batch):
        x_img = batch[self.hparams.image_key]   # (B, 2, H, W) — image channels
        mask  = batch[self.hparams.mask_key]    # (B, 1, H, W) — Membrane masks

        # concatenate mask as 3rd input channel
        x_in  = torch.cat([x_img, mask.float()], dim=1).to(torch.double)   # (B, 3, H, W)
        recon, z, mu, logvar = self.vae(x_in)
        # loss on image channels only (first nc_img), inside pCellmask
        # m = (mask > 0).expand_as(x_img)                   # (B, 2, H, W)
        recon_loss = self.recon_function(recon, x_in)

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
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
