"""A small, self-contained tutorial on training with PyTorch Lightning.

Normally, training a model in PyTorch means writing a loop by hand: loop over
batches, call ``loss.backward()`` and ``optimizer.step()``, remember to move data
to the GPU, then write a second loop for validation... PyTorch Lightning takes
care of all of that. You just describe *what* to do, and the ``Trainer`` runs the
loop *for* you.

There are three pieces to learn:

1. A ``Dataset`` / ``DataLoader`` -- where the data comes from.
2. A ``LightningModule``          -- your model plus how one training step works.
3. The ``Trainer``                -- runs the whole training loop.

How to run this file
---------------------
* As a script:  ``python example/lightning_tutorial.py``
* Or open it in VS Code / Jupyter and run each ``# %%`` cell from top to bottom.
"""

# %% [markdown]
# # 0. Imports
# ``lightning.pytorch`` is the Lightning API. ``torch`` is plain PyTorch.

# %%
import torch
from lightning.pytorch import LightningModule, Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# # 1. The data
#
# A ``Dataset`` only has to answer two questions:
#
# * ``__len__``     -> how many examples are there?
# * ``__getitem__`` -> give me example number ``i``.
#
# To keep the tutorial offline, we make up random 8x8 grayscale "images".
# Later, you would replace this class with one that loads your real data.


# %%
class RandomImageDataset(Dataset):
    """A dataset of random images, each shaped (1, 8, 8)."""

    def __init__(self, n_samples: int = 512, seed: int = 0) -> None:
        # A seeded generator means everyone gets the same "random" images.
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n_samples, 1, 8, 8, generator=generator)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        return self.images[index]


# A DataLoader wraps a Dataset and serves it in batches (here, 32 at a time).
train_loader = DataLoader(RandomImageDataset(seed=0), batch_size=32, shuffle=True)
val_loader = DataLoader(RandomImageDataset(n_samples=128, seed=1), batch_size=32)

# %% [markdown]
# # 2. The model: a tiny autoencoder
#
# An autoencoder squeezes each image down to a few numbers (the *encoder*) and
# then tries to rebuild the original image from them (the *decoder*). We train it
# so the rebuilt image looks like the input.
#
# As a ``LightningModule`` we implement four methods:
#
# * ``forward``              -- run data through the network
# * ``training_step``        -- compute the loss for one batch of training data
# * ``validation_step``      -- the same, but on validation data
# * ``configure_optimizers`` -- which optimizer updates the weights


# %%
class AutoEncoder(LightningModule):
    """A minimal autoencoder for 1x8x8 images."""

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        # 64 = 1 * 8 * 8 pixels flattened into a vector.
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(64, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        latent = self.encoder(x)
        rebuilt = self.decoder(latent)
        # Reshape the flat vector back into an image.
        return rebuilt.view(-1, 1, 8, 8)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        reconstruction = self(batch)
        # How different is the rebuilt image from the original?
        loss = nn.functional.mse_loss(reconstruction, batch)
        # self.log records the value; prog_bar=True shows it in the progress bar.
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        reconstruction = self(batch)
        loss = nn.functional.mse_loss(reconstruction, batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Adam is a good default optimizer. lr is the learning rate (step size).
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# %% [markdown]
# # 3. Train it
#
# The ``Trainer`` runs the loop. ``accelerator="auto"`` uses your GPU (or Apple
# MPS) if you have one, otherwise the CPU -- you do not change any code to switch.
#
# We use ``max_epochs=3`` to keep it quick; raise it for real training.

# %%
model = AutoEncoder()
trainer = Trainer(max_epochs=3, accelerator="auto", log_every_n_steps=5)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# %% [markdown]
# # What next?
#
# * Replace ``RandomImageDataset`` with one that loads your real images.
# * Make the network bigger, or turn the autoencoder into a **VAE** (the "VAEder"
#   in this repo!) by having the encoder predict a mean and a variance and adding
#   a KL-divergence term to the loss.
# * For larger projects, move the DataLoaders into a ``LightningDataModule`` so
#   all the data setup lives in one tidy, reusable place.
