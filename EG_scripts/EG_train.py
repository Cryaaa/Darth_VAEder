"""VAE training entry point.

Usage
-----
    python "/mnt/efs/dl_jrc/student_data/S-EG/project/Darth_VAEder/EG_scripts/EG_train.py" \
        --zarr  /mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr \
        --table /mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv \
        --out  /mnt/efs/dl_jrc/student_data/S-EG/project/VAE1  \
        --epochs 75
        --z-dim 50
        --beta 1

Logs (TensorBoard + CSV) → <out>/logs/vae/
Checkpoints              → <out>/checkpoints/
    last.ckpt   — always overwritten at end of each epoch (resumability)
    best.ckpt   — best val/loss seen so far

TensorBoard:
    tensorboard --logdir outputs/logs
"""

import argparse
from pathlib import Path

import torch
from torch.nn import MSELoss
from torchvision.utils import make_grid
from torchvision.transforms import v2, GaussianBlur

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from darth_vaeder.datamodules.dataset_EG import BCDataModule, percentile_norm, BorderCellDataset, no_transform
from darth_vaeder.models import LitVAE


# ── Reconstruction visualisation callback ─────────────────────────────────────

class ReconVizCallback(Callback):
    """Log a fixed grid of input vs reconstructed patches to TensorBoard.

    A small batch from the val set is grabbed once at fit-start and reused
    every `every_n_epochs` epochs, so comparisons are consistent across time.
    """

    def __init__(self, n_cells: int = 8, every_n_epochs: int = 5):
        self.n_cells       = n_cells
        self.every_n_epochs = every_n_epochs
        self._batch        = None

    def on_fit_start(self, trainer, pl_module):
        loader      = trainer.datamodule.val_dataloader()
        self._batch = next(iter(loader))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if self._batch is None:
            return

        tb = next(
            (l for l in trainer.loggers if isinstance(l, TensorBoardLogger)), None
        )
        if tb is None:
            return

        x_img = self._batch[pl_module.hparams.image_key][: self.n_cells].to(pl_module.device)
        mask  = self._batch[pl_module.hparams.mask_key][: self.n_cells].to(pl_module.device)
        x_in  = torch.cat([x_img, mask.float()], dim=1)  # same 3ch input as _step

        pl_module.eval()
        with torch.no_grad():
            recon, *_ = pl_module.vae(x_in)
        pl_module.train()

        x_img, recon = x_img.cpu(), recon.cpu()

        def _norm(t: torch.Tensor) -> torch.Tensor:
            """Per-image min-max normalise to [0, 1] for display."""
            mn = t.flatten(1).min(1).values[:, None, None, None]
            mx = t.flatten(1).max(1).values[:, None, None, None]
            return (t - mn) / (mx - mn + 1e-6)

        ch_names = ["membrane", "nuclei"]
        for ch, name in enumerate(ch_names):
            inp = _norm(x_img[:, ch : ch + 1])   # (N, 1, H, W) — image channels only
            rec = _norm(recon[:, ch : ch + 1])

            # interleave: [inp0, rec0, inp1, rec1, …] — each row = one cell
            pairs = torch.stack([inp, rec], dim=1).view(-1, 1, *inp.shape[-2:])
            grid  = make_grid(pairs, nrow=2, pad_value=0.5)
            tb.experiment.add_image(
                f"recon/{name}", grid, global_step=trainer.current_epoch
            )


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # paths
    p.add_argument("--zarr",    required=True, help="Path to multinucleation.zarr")
    p.add_argument("--table",   required=True, help="Path to cell_table.csv")
    p.add_argument("--out",     default="outputs", help="Root dir for checkpoints + logs")
    p.add_argument("--splits",  default=None,
                   help="Pre-saved splits.json (skips auto-split if provided)")
    # data
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--workers", type=int,   default=7)
    # model
    p.add_argument("--nc",      type=int,   default=3,   help="Input channels to encoder (2 image + 1 mask)")
    p.add_argument("--z-dim",   type=int,   default=10,  help="Latent dimensionality")
    p.add_argument("--beta",    type=float, default=0.0, help="KL weight; 0 = pure reconstruction")
    p.add_argument("--lr",      type=float, default=1e-4)
    # training
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--devices",     type=int, default=1)
    p.add_argument("--patience",    type=int, default=0,
                   help="Early stopping patience on val/loss; 0 = disabled")
    p.add_argument("--viz-every",   type=int, default=5,
                   help="Log reconstruction images every N epochs")
    p.add_argument("--viz-cells",   type=int, default=8,
                   help="Number of val cells shown in reconstruction grid")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out  = Path(args.out)

    dataset = BorderCellDataset(
    args.table, 
    zarr_path = args.zarr, input_array_name="max_projection", 
    input_mask_name="3D_mask_corrected", normalization_function= percentile_norm,
    )

    # spatial transforms = 
    dm = BCDataModule(dataset, 
                      spatial_transforms=v2.Compose([v2.RandomRotation(180), 
                                                    v2.RandomHorizontalFlip(p=1),
                                                    v2.RandomVerticalFlip(p=1), v2.RandomAffine(15), 
                                                    v2.RandomErasing(p=1, scale=(0.02,0.33))]), intensity_transforms= GaussianBlur(kernel_size=3, sigma=(0.1,2)), batch_size=args.batch, num_workers = args.workers)

    # ── model ─────────────────────────────────────────────────────────────
    model = LitVAE(
        nc=args.nc,
        image_key = 'source',
        mask_key = 'masks',
        z_dim=args.z_dim,
        beta=args.beta,
        lr=args.lr,
    )

    # ── callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        # saves the single best checkpoint by val/loss
        ModelCheckpoint(
            #dirpath=out / "checkpoints",
            #filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,      # last.ckpt updated every epoch
        ),
        ReconVizCallback(
            n_cells=args.viz_cells,
            every_n_epochs=args.viz_every,
        ),
    ]
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")
        )

    # ── loggers ───────────────────────────────────────────────────────────
    log_dir = out / "logs"
    loggers = [
        TensorBoardLogger(log_dir, name="vae"),
        CSVLogger(log_dir, name="vae"),          # plain-text backup
    ]

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val = 0.5,
        fast_dev_run=False,   # set True for debugging (runs 1 batch only
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()



# dataset = BorderCellDataset(
#     "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv", 
#     "/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr", "max_projection", 
#     "3D_mask_corrected"
#     )

# train_module = BCDataModule(dataset, percentile_norm, None, batch_size = 4)
# model = LitVAE(nc=3, image_key="source", mask_key = "masks", beta = 0)

