"""VAE training entry point.

Usage
-----
    python "Joaquin'scripts/train.py" \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv \
        --out   outputs \
        --epochs 50

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
from torchvision.utils import make_grid

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from darth_vaeder.datamodules import MultinucDataModule
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

        x_img    = self._batch[pl_module.hparams.image_key][: self.n_cells].to(pl_module.device)
        mask     = self._batch[pl_module.hparams.mask_key][: self.n_cells].to(pl_module.device)
        nuc_mask = self._batch[pl_module.hparams.nuc_mask_key][: self.n_cells].to(pl_module.device)
        x_in  = torch.cat([x_img, mask.float(), nuc_mask.float()], dim=1)  # same 4ch input as _step

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
    p.add_argument("--splits",    default=None,
                   help="Pre-saved splits.json (skips auto-split if provided)")
    p.add_argument("--warm-ckpt", default=None,
                   help="Checkpoint to warm-start from.  Weights are loaded; "
                        "optimizer and epoch counter reset (use for new training "
                        "phases, e.g. switching beta).  Architecture must match.")
    # data
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--workers", type=int,   default=7)
    # model
    p.add_argument("--nc",      type=int,   default=4,   help="Input channels to encoder (2 image + 2 masks)")
    p.add_argument("--z-dim",   type=int,   default=10,  help="Latent dimensionality")
    p.add_argument("--beta",    type=float, default=0.0, help="KL weight; 0 = pure reconstruction")
    p.add_argument("--lr",      type=float, default=1e-3)
    # training
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--devices",     type=int, default=1)
    p.add_argument("--patience",    type=int, default=0,
                   help="Early stopping patience on val/loss; 0 = disabled")
    p.add_argument("--viz-every",   type=int, default=1,
                   help="Log reconstruction images every N epochs")
    p.add_argument("--viz-cells",   type=int, default=8,
                   help="Number of val cells shown in reconstruction grid")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────
    dm = MultinucDataModule(
        data_path=args.zarr,
        cell_table_csv=args.table,
        channels=(0, 1),
        batch_size=args.batch,
        num_workers=args.workers,
        augment=True,
    )
    if args.splits:
        dm.load_splits(args.splits)

    # ── model ─────────────────────────────────────────────────────────────
    if args.warm_ckpt:
        # Load pre-trained weights but apply current hyperparameters.
        # Optimizer resets → correct for new training phases (e.g. adding KL).
        model = LitVAE.load_from_checkpoint(
            args.warm_ckpt,
            nc=args.nc, z_dim=args.z_dim, beta=args.beta, lr=args.lr,
        )
        print(f"  warm start  : {args.warm_ckpt}  (beta={args.beta})")
    else:
        model = LitVAE(nc=args.nc, z_dim=args.z_dim, beta=args.beta, lr=args.lr)

    # ── loggers ───────────────────────────────────────────────────────────
    # TB logger is created first; its auto-incremented version is then reused
    # for the CSV logger and checkpoint subdir so all run outputs share one N:
    #   logs/vae/version_N/       ← TensorBoard + CSV
    #   checkpoints/version_N/    ← best.ckpt + last.ckpt
    log_dir   = out / "logs"
    tb_logger = TensorBoardLogger(log_dir, name="vae")
    version   = tb_logger.version          # int, e.g. 11
    ckpt_dir  = out / "checkpoints" / f"version_{version}"
    print(f"  run version : {version}")
    print(f"  logs        : {log_dir}/vae/version_{version}/")
    print(f"  checkpoints : {ckpt_dir}/")
    loggers = [
        tb_logger,
        CSVLogger(log_dir, name="vae", version=version),  # matches TB version
    ]

    # ── callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        # versioned dir matches log version; epoch-stamped names for top-5
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=5,
            save_last=True,      # last.ckpt updated every epoch
            auto_insert_metric_name=False,
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

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
        fast_dev_run=False,   # set True for debugging (runs 1 batch only
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
