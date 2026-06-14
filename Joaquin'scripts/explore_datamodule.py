"""Quick sanity-check and visualization for the data pipeline.

Instantiates the datamodule, runs one train batch, and plots a grid of cells:
    row 0 — membrane channel
    row 1 — nuclei channel
    row 2 — cell mask overlay on membrane

Usage (server)
--------------
    python scripts/explore_datamodule.py \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table /mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/outputs/cell_table.csv \
        --out   outputs/datamodule_grid.png \
        --n     16
"""

import argparse

import matplotlib
matplotlib.use("Agg")   # headless-safe; switch to "TkAgg" / "MacOSX" for interactive
import matplotlib.pyplot as plt
import numpy as np
import torch

from darth_vaeder.datamodules import MultinucDataModule


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",  required=True, help="Path to multinucleation.zarr")
    p.add_argument("--table", required=True, help="Path to cell_table.csv")
    p.add_argument("--out",   default="outputs/datamodule_grid.png",
                   help="Output image path")
    p.add_argument("--n",     type=int, default=16,
                   help="Number of cells to plot (should be a perfect square or 4×N)")
    p.add_argument("--patch-type", default="cPatches",
                   choices=["cPatches", "bbPatches"])
    p.add_argument("--split", default="train",
                   choices=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, easier to debug)")
    return p.parse_args()


def plot_batch(batch: dict, n: int, out_path: str):
    images = batch["image"][:n]         # (N, C, H, W) in [0, 1]
    masks  = batch["cCellmask"][:n]     # (N, 1, H, W) int labels

    ncols = max(1, int(np.ceil(np.sqrt(n))))
    nrows_per_strip = 3
    fig, axes = plt.subplots(
        nrows_per_strip * ((n + ncols - 1) // ncols),
        ncols,
        figsize=(ncols * 2.5, nrows_per_strip * ((n + ncols - 1) // ncols) * 2.5),
    )
    axes = np.array(axes).reshape(-1, ncols)

    grid_row = 0
    for i in range(n):
        col = i % ncols
        if i > 0 and col == 0:
            grid_row += nrows_per_strip

        img = images[i].cpu().numpy()           # (C, H, W)
        msk = masks[i, 0].cpu().numpy() > 0    # (H, W) bool

        mem  = img[0]
        nuc  = img[1] if img.shape[0] > 1 else np.zeros_like(mem)
        overlay = np.stack([mem, mem, mem], axis=-1)
        overlay[msk, 0] = np.clip(overlay[msk, 0] + 0.3, 0, 1)    # red tint in-mask

        axes[grid_row,     col].imshow(mem,  cmap="gray", vmin=0, vmax=1)
        axes[grid_row + 1, col].imshow(nuc,  cmap="gray", vmin=0, vmax=1)
        axes[grid_row + 2, col].imshow(overlay)

        cond = batch["metadata"]["condition"][i]
        rep  = batch["metadata"]["replicate"][i]
        axes[grid_row, col].set_title(f"{rep}/{cond}", fontsize=7)

    # row labels on leftmost column
    for strip in range((n + ncols - 1) // ncols):
        r0 = strip * nrows_per_strip
        axes[r0,     0].set_ylabel("membrane", fontsize=8)
        axes[r0 + 1, 0].set_ylabel("nuclei",   fontsize=8)
        axes[r0 + 2, 0].set_ylabel("mask ovl", fontsize=8)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved → {out_path}")


def main():
    args = parse_args()

    dm = MultinucDataModule(
        data_path=args.zarr,
        cell_table_csv=args.table,
        patch_type=args.patch_type,
        channels=(0, 1),
        masks=("cCellmask",),       # returned per sample for mask overlay
        batch_size=args.n,
        num_workers=args.workers,
        persistent_workers=False,   # single batch → no benefit in keeping workers
        augment=False,              # inspect raw normalised patches, no augmentation
    )
    dm.setup("fit")

    loader_map = {"train": dm.train_dataloader,
                  "val":   dm.val_dataloader,
                  "test":  dm.test_dataloader}
    loader = loader_map[args.split]()
    batch  = next(iter(loader))

    n_got = batch["image"].shape[0]
    print(f"Batch shape : {tuple(batch['image'].shape)}")
    print(f"Image range : [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"Conditions  : {sorted(set(batch['metadata']['condition']))}")

    split_sizes = {k: len(v) for k, v in dm.splits.items()}
    print(f"Split sizes : { {k: f'{v:,}' for k, v in split_sizes.items()} }")

    plot_batch(batch, min(args.n, n_got), args.out)


if __name__ == "__main__":
    main()
