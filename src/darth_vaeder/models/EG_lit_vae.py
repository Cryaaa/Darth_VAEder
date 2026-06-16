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
                  nc: int = 3,
                  nc_img: int=2,
                  image_key: str = "cPatch",
                  mask_key: str = "pCellmask",
                  target_key: str = "target",
                  recon_function = F.mse_loss,
                  z_dim: int = 64,
                  beta: float = 1.0,
                  lr: float = 1e-3):
        super().__init__()
        self.recon_function = recon_function
        self.nc_img = nc_img
        self.save_hyperparameters()
        self.vae  = VAEResNet18(nc=nc, z_dim=z_dim)
        self.image_key = image_key
        self.mask_key = mask_key
        self.target_key = target_key
        self.beta = beta
        self.lr   = lr

    def forward(self, x):
        """x: (B, nc, H, W) → recon, z, mu, logvar"""
        return self.vae(x)

    def _step(self, batch):
        x    = batch[self.image_key]       # (B, nc, H, W)  normalised, bg=0
        # print(f"x shape is {x.shape, x.type()}")
        mask = batch[self.mask_key]    # (B, 1, H, W)   dilated crop mask
        # print(f"mask shape is {mask.shape, mask.type()}")
        x_in  = torch.cat([x, mask], dim=1).to(torch.float)   # (B, 3, H, W)
        target = torch.cat([(batch[self.target_key]), mask], dim=1).to(torch.float)
        # print(f"x_in shape is {x_in.shape, x_in.type()}")
        recon, z, mu, logvar = self.vae(x_in)


        # masked reconstruction: MSE averaged over in-mask pixels only
        # mask is (B,1,H,W); expand to match (B,nc,H,W) for indexing
        #m = (mask > 0).expand_as(x)
        #recon_loss = self.recon_function(recon[m], x[m])
        recon_loss = self.recon_function(recon, target, reduction='sum')/recon.shape[0]

        # KL divergence: mu/logvar are (B, z_dim, H', W') for this spatial VAE
        # -0.5 * sum(1 + log_sigma^2 - mu^2 - sigma^2) averaged over all dims
        #kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
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
