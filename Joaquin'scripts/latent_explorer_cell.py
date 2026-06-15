"""Cell-friendly latent explorer utilities.

Designed to be imported into a notebook or run inside a long-running Python
cell. All prints use `flush=True` so output appears live in the cell.

Usage (in a notebook cell):
  from Joaquin'scripts.latent_explorer_cell import run_explorer
  run_explorer(ckpt=..., zarr=..., table=..., out='outputs/latent_cell', split='val',
               batch=8, workers=0, device='cpu')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from darth_vaeder.datamodules import MultinucDataModule
from darth_vaeder.datamodules.JS_zarr_datamodule import vae_collate
from darth_vaeder.models import LitVAE


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo + 1e-6)


def build_loader(dm: MultinucDataModule, split: str, batch_size: int, workers: int):
    dm.setup()
    datasets = {
        "train": dm.train_dataset,
        "val":   dm.val_dataset,
        "test":  dm.test_dataset,
    }
    if split == "all":
        ds = ConcatDataset([datasets["train"], datasets["val"], datasets["test"]])
    else:
        ds = datasets[split]
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, collate_fn=vae_collate, drop_last=False)


@torch.no_grad()
def encode_dataset(model: LitVAE, loader, device, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval().to(device)
    image_key = model.hparams.image_key
    mask_key  = model.hparams.mask_key
    nc_img    = model.nc_img

    Z, errs, meta_rows = [], [], []
    gi = 0
    n_batches = len(loader)

    for bi, batch in enumerate(loader):
        if verbose:
            print(f"[encode] batch {bi+1}/{n_batches}", flush=True)
        x_img = batch[image_key].to(device)
        mask  = batch[mask_key].to(device)
        x_in  = torch.cat([x_img, mask.float()], dim=1)

        with torch.no_grad():
            recon, _z, mu, _logvar = model.vae(x_in)

        m2    = (mask > 0).expand_as(x_img).float()
        diff2 = (recon[:, :nc_img] - x_img) ** 2
        err   = (diff2 * m2).sum(dim=[1, 2, 3]) / m2.sum(dim=[1, 2, 3]).clamp_min(1)

        Z.append(mu.cpu().numpy())
        errs.append(err.cpu().numpy())

        md = batch["metadata"]
        bsz = x_img.shape[0]
        for j in range(bsz):
            meta_rows.append({k: md[k][j] for k in md})

        gi += bsz

    Z         = np.concatenate(Z, axis=0)
    recon_err = np.concatenate(errs, axis=0)
    meta = pd.DataFrame(meta_rows)
    meta["recon_err"] = recon_err
    if verbose:
        print(f"[encode] done — {Z.shape[0]} cells encoded", flush=True)
    return Z, recon_err, meta


def reduce_dims(Z: np.ndarray, n_components: int = 2):
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        return pca.fit_transform(Z)
    except Exception:
        Zc = Z - Z.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        return (U[:, :n_components] * S[:n_components])


def plot_embedding_2d(emb, meta, out_dir: Path):
    plt.figure(figsize=(6, 5))
    if 'condition' in meta.columns:
        groups = meta['condition'].astype(str).unique()
        for g in groups:
            sel = meta['condition'].astype(str) == g
            plt.scatter(emb[sel, 0], emb[sel, 1], s=8, alpha=0.6, label=str(g))
        plt.legend(markerscale=2)
    else:
        plt.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.6)
    plt.title("Latent space (2D) — PCA")
    plt.xlabel("dim0"); plt.ylabel("dim1")
    out = out_dir / "embedding_2d.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[plot] wrote {out}", flush=True)


@torch.no_grad()
def plot_reconstructions(model, loader, device, out_dir: Path, recon_err, n=6):
    image_key = model.hparams.image_key
    mask_key  = model.hparams.mask_key

    order_best  = np.argsort(recon_err)[:n]
    order_worst = np.argsort(recon_err)[-n:]
    rng = np.random.default_rng(1)
    order_rand  = rng.choice(len(recon_err), size=n, replace=False)
    wanted = {int(i): None for i in [*order_best, *order_worst, *order_rand]}

    gi = 0
    for batch in loader:
        x_img = batch[image_key]
        mask  = batch[mask_key]
        for j in range(x_img.shape[0]):
            if gi + j in wanted:
                wanted[gi + j] = (x_img[j].clone(), mask[j].clone())
        gi += x_img.shape[0]
        if all(v is not None for v in wanted.values()):
            break

    groups = [("best", order_best), ("worst", order_worst), ("random", order_rand)]
    fig, axes = plt.subplots(len(groups) * 2, n, figsize=(n * 1.6, len(groups) * 3.4))
    for gr, (name, order) in enumerate(groups):
        for k, idx in enumerate(order):
            x_img, mask = wanted[int(idx)]
            x_in = torch.cat([x_img, mask.float()], dim=0).unsqueeze(0).to(device)
            recon, *_ = model.vae(x_in)
            inp = _norm01(x_img[0].numpy())
            rec = _norm01(recon[0, 0].cpu().numpy())
            axes[gr * 2,     k].imshow(inp, cmap="gray"); axes[gr * 2,     k].axis("off")
            axes[gr * 2 + 1, k].imshow(rec, cmap="gray"); axes[gr * 2 + 1, k].axis("off")
            if k == 0:
                axes[gr * 2,     k].set_ylabel(f"{name}\ninput", fontsize=8)
                axes[gr * 2 + 1, k].set_ylabel("recon",          fontsize=8)
            axes[gr * 2, k].set_title(f"err={recon_err[idx]:.3f}", fontsize=7)
    fig.suptitle("Reconstructions (membrane channel): input vs decoded", y=1.005)
    fig.tight_layout()
    out = out_dir / "reconstructions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}", flush=True)


def run_explorer(ckpt: str, zarr: str, table: str, out: str = "outputs/latent_cell",
                 split: str = "val", batch: int = 8, workers: int = 0, device: str = "auto",
                 verbose: bool = True):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _pick_device(device)
    if verbose:
        print(f"[run] loading checkpoint ({dev})", flush=True)
    model = LitVAE.load_from_checkpoint(ckpt, map_location=dev)
    if verbose:
        print(f"[run] z_dim={model.hparams.z_dim} nc={model.hparams.nc}", flush=True)

    dm = MultinucDataModule(data_path=zarr, cell_table_csv=table,
                            channels=(0, 1), batch_size=batch, num_workers=workers,
                            augment=False)
    loader = build_loader(dm, split, batch, workers)
    n_total = len(loader.dataset)
    print(f"[run] split='{split}' cells={n_total}", flush=True)

    Z, recon_err, meta = encode_dataset(model, loader, dev, verbose=verbose)
    meta.to_csv(out_dir / "latents.csv", index=False)
    np.savez(out_dir / "latents.npz", Z=Z, recon_err=recon_err,
             cell_idx=meta.get("cell_idx", pd.Series(range(len(Z)))).to_numpy())
    print(f"[run] wrote latents to {out_dir}", flush=True)

    emb2 = reduce_dims(Z, n_components=2)
    plot_embedding_2d(emb2, meta, out_dir)
    plot_reconstructions(model, loader, dev, out_dir, recon_err)

    print(f"[run] done. outputs in {out_dir.resolve()}", flush=True)
    return dict(Z=Z, recon_err=recon_err, meta=meta, out_dir=str(out_dir.resolve()))
