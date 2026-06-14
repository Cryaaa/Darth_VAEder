"""VAE training entry point.

Usage
-----
    python scripts/train.py \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv \
        --out   outputs

First run: cPatch (2ch) + cCellmask only (include_bb disabled by default).
Checkpoints → <out>/checkpoints/   Logs → <out>/logs/
"""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from darth_vaeder.datamodules import MultinucDataModule
from darth_vaeder.models import LitVAE


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
    p.add_argument("--batch",   type=int,   default=64)
    p.add_argument("--workers", type=int,   default=8)
    p.add_argument("--include-bb", action="store_true",
                   help="Also feed bbPatch (context window) to the model")
    # model
    p.add_argument("--nc",      type=int,   default=2,   help="Input channels (membrane+nuclei)")
    p.add_argument("--z-dim",   type=int,   default=64,  help="Latent dimensionality")
    p.add_argument("--beta",    type=float, default=1.0, help="KL weight (beta-VAE)")
    p.add_argument("--lr",      type=float, default=1e-3)
    # training
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--devices", type=int,   default=1)
    p.add_argument("--patience",type=int,   default=15,
                   help="Early stopping patience (val/loss); 0 to disable")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out)

    # ── data ──────────────────────────────────────────────────────────────
    dm = MultinucDataModule(
        data_path=args.zarr,
        cell_table_csv=args.table,
        channels=(0, 1),
        include_bb=args.include_bb,
        batch_size=args.batch,
        num_workers=args.workers,
        augment=True,
    )
    if args.splits:
        dm.load_splits(args.splits)

    # ── model ─────────────────────────────────────────────────────────────
    model = LitVAE(nc=args.nc, z_dim=args.z_dim, beta=args.beta, lr=args.lr)

    # ── callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=out / "checkpoints",
            filename="vae-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
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
        logger=CSVLogger(out / "logs", name="vae"),
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
