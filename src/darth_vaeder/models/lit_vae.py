"""LightningModule that wraps VAEResNet18 for training with PyTorch Lightning.

Loss = masked reconstruction (MSE inside cCellmask) + beta * KL divergence.
Masking ensures the VAE is scored only on cell pixels, not the zero background.
Logs train/val loss, reconstruction, and KL terms separately.
"""

import lightning as L


class LitVAE(L.LightningModule):
    """Lightning wrapper for VAEResNet18.

    Parameters
    ----------
    nc:
        Number of input channels (2: membrane + nuclei).
    z_dim:
        Latent space dimensionality.
    beta:
        Weight on the KL divergence term (beta-VAE).
    lr:
        Learning rate.
    """

    def __init__(self, nc: int = 2, z_dim: int = 64, beta: float = 1.0, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
