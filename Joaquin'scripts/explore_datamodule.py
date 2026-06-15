"""Explore cnPatches at three pipeline stages for the same N cells.

Produces a single grid PNG with 9 row-bands per N cells:

    Stage 1 — RAW        membrane | nuclei | overlay (pCellmask=red, pNucmask=blue)
    Stage 2 — NORMALIZED after NormalizeFromStats (precomputed p1/p99 per channel)
    Stage 3 — AUGMENTED  after full training pipeline (normalize + rotate + flip)

Usage
-----
    python "Joaquin'scripts/explore_datamodule.py" \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv \
        --out   outputs/explore_grid.png \
        --n     8 --split val
"""

import argparse
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from darth_vaeder.datamodules import MultinucDataModule
from darth_vaeder.datamodules.JS_transforms import NormalizeFromStats, build_train_transforms


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",    required=True)
    p.add_argument("--table",   required=True)
    p.add_argument("--out",     default="outputs/explore_grid.png")
    p.add_argument("--n",       type=int, default=8)
    p.add_argument("--split",   default="val", choices=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def _norm_display(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _overlay(mem, cell_mask, nuc_mask):
    g = _norm_display(mem)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[cell_mask > 0, 0] = np.clip(rgb[cell_mask > 0, 0] + 0.35, 0, 1)
    rgb[nuc_mask  > 0, 2] = np.clip(rgb[nuc_mask  > 0, 2] + 0.45, 0, 1)
    return rgb


def _show(ax, img, **kwargs):
    ax.imshow(img, **kwargs)
    ax.set_xticks([]); ax.set_yticks([])


def load_pnucmask(zarr_root, sample):
    md  = sample["metadata"]
    rep, cond, img = md["replicate"], md["condition"], md["image_name"]
    loc = int(md["local_cell_index"])
    pg  = zarr_root[f"patches/{rep}/{cond}/{img}"]
    key = "pNucmask" if "pNucmask" in pg else "cNucmask"
    return np.ascontiguousarray(pg[key][loc])


def main():
    args = parse_args()

    dm = MultinucDataModule(
        data_path=args.zarr,
        cell_table_csv=args.table,
        channels=(0, 1),
        batch_size=1,
        num_workers=args.workers,
        persistent_workers=False,
        augment=False,
    )
    dm.setup("fit" if args.split in ("train", "val") else "test")

    dataset = {"train": dm.train_dataset,
               "val":   dm.val_dataset,
               "test":  dm.test_dataset}[args.split]

    real_transform    = dataset.transform
    dataset.transform = None
    n = min(args.n, len(dataset))
    raw_samples = [dataset[i] for i in range(n)]
    dataset.transform = real_transform

    norm_only = NormalizeFromStats()
    full_aug  = build_train_transforms(image_keys=("cPatch",),
                                       mask_keys=("pCellmask", "pNucmask"))
    zroot = zarr.open_group(args.zarr, mode="r")

    conds = [s["metadata"]["condition"] for s in raw_samples]
    print(f"Split: {args.split}  cells: {n}  conditions: {sorted(set(conds))}")
    raw0 = raw_samples[0]["cPatch"]
    print(f"Raw range: [{raw0.min():.3f}, {raw0.max():.3f}]")
    normed0 = norm_only(copy.deepcopy(raw_samples[0]))["cPatch"]
    print(f"Normed range: [{normed0.min():.3f}, {normed0.max():.3f}]")

    STAGES       = ["RAW", "NORMALIZED", "AUGMENTED"]
    ROWS_PER_STG = 3
    n_rows       = len(STAGES) * ROWS_PER_STG

    fig, axes = plt.subplots(n_rows, n, figsize=(n * 2.5, n_rows * 2.5))
    axes = np.array(axes).reshape(n_rows, n)

    for col, raw in enumerate(raw_samples):
        normed     = norm_only(copy.deepcopy(raw))
        cell_mask  = raw["pCellmask"][0].numpy()
        nuc_mask   = load_pnucmask(zroot, raw)
        nuc_mask_t = torch.from_numpy(nuc_mask).long().unsqueeze(0)
        raw_aug    = copy.deepcopy(raw)
        raw_aug["pNucmask"] = nuc_mask_t
        augd = full_aug(raw_aug)

        for si, (stage_name, sample) in enumerate(zip(STAGES, [raw, normed, augd])):
            base_row = si * ROWS_PER_STG
            img  = sample["cPatch"].numpy()
            mem  = img[0]
            nuc  = img[1]
            s_cell = sample["pCellmask"][0].numpy()
            s_nuc  = sample["pNucmask"][0].numpy() if "pNucmask" in sample else nuc_mask

            _show(axes[base_row,     col], _norm_display(mem), cmap="gray", vmin=0, vmax=1)
            _show(axes[base_row + 1, col], _norm_display(nuc), cmap="gray", vmin=0, vmax=1)
            _show(axes[base_row + 2, col], _overlay(mem, s_cell, s_nuc))

            if col == 0:
                axes[base_row,     0].set_ylabel(f"{stage_name}\nmembrane", fontsize=8)
                axes[base_row + 1, 0].set_ylabel("nuclei",  fontsize=8)
                axes[base_row + 2, 0].set_ylabel("overlay", fontsize=8)

        axes[0, col].set_title(
            f"{raw['metadata']['replicate']}\n{raw['metadata']['condition']}", fontsize=7)

    stage_colors = ["#f0f4ff", "#fff4e0", "#f0ffe0"]
    for si, color in enumerate(stage_colors):
        for r in range(ROWS_PER_STG):
            axes[si * ROWS_PER_STG + r, 0].set_facecolor(color)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()