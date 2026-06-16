"""Latent-space explorer for the multinucleation VAE.

All outputs are PNG files saved to --out.  No HTML, no browser needed.

Outputs
-------
    latents.csv / latents.npz        raw per-cell latents (mu) + metadata
    umap_condition.png               UMAP coloured by condition
    umap_replicate.png               UMAP coloured by replicate
    umap_recon_err.png               UMAP coloured by reconstruction error
    traversals.png                   each latent dim walked -3σ → +3σ
    interpolation.png                straight path between two cells in latent space
    reconstructions.png              input vs recon for random / best / worst cells

Usage (server, darth-vaeder env)
---------------------------------
    python "Joaquin'scripts/latent_explorer.py" \
        --ckpt  outputs/checkpoints/best.ckpt \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv \
        --out   outputs/latent \
        --split test

Requires: umap-learn  (pip install umap-learn)
          matplotlib, numpy, pandas, torch  (already in env)
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from darth_vaeder.datamodules import MultinucDataModule
from darth_vaeder.datamodules.JS_zarr_datamodule import vae_collate
from darth_vaeder.JS_models import LitVAE


# ── Utilities ─────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def _pick_device(req):
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-6)


def _section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ── Data ──────────────────────────────────────────────────────────────────────

def build_loader(dm: MultinucDataModule, split: str, batch_size: int, workers: int):
    dm.setup()
    datasets = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}
    ds = ConcatDataset(list(datasets.values())) if split == "all" else datasets[split]
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, collate_fn=vae_collate, drop_last=False)


def _build_x_in(batch, model, device):
    """Build the 4-channel encoder input matching _step in lit_vae.py."""
    x_img    = batch[model.hparams.image_key].to(device)
    mask     = batch[model.hparams.mask_key].to(device)
    nuc_mask = batch[model.hparams.nuc_mask_key].to(device)
    return torch.cat([x_img, mask.float(), nuc_mask.float()], dim=1)


# ── Step 1: encode ────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_dataset(model: LitVAE, loader, device):
    model.eval().to(device)
    nc_img = model.nc_img

    Z, errs, meta_rows = [], [], []
    gi = 0
    n_batches = len(loader)

    for bi, batch in enumerate(loader):
        x_in = _build_x_in(batch, model, device)
        with _silence():
            recon, _z, mu, _logvar = model.vae(x_in)

        mask  = batch[model.hparams.mask_key].to(device)
        x_img = batch[model.hparams.image_key].to(device)
        m2    = (mask > 0).expand_as(x_img).float()
        diff2 = (recon[:, :nc_img] - x_img) ** 2
        err   = (diff2 * m2).sum(dim=[1, 2, 3]) / m2.sum(dim=[1, 2, 3]).clamp_min(1)

        Z.append(mu.cpu().numpy())
        errs.append(err.cpu().numpy())

        md  = batch["metadata"]
        bsz = x_img.shape[0]
        for j in range(bsz):
            meta_rows.append({k: md[k][j] for k in md})
        gi += bsz

        if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
            print(f"    batch {bi+1}/{n_batches}  ({gi} cells)")

    Z         = np.concatenate(Z, axis=0)
    recon_err = np.concatenate(errs, axis=0)
    meta      = pd.DataFrame(meta_rows)
    meta["recon_err"] = recon_err
    return Z, recon_err, meta


# ── Step 2: UMAP ──────────────────────────────────────────────────────────────

def compute_umap(Z: np.ndarray, seed: int = 42) -> np.ndarray:
    try:
        import umap
        print("    running UMAP …")
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
    except ImportError:
        print("    [warn] umap-learn not found, falling back to PCA. "
              "pip install umap-learn")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(Z)


def _scatter_png(emb, values, title, cbar_label, out_path, cmap="tab10", categorical=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    if categorical:
        uniq = sorted(set(str(v) for v in values))
        palette = plt.get_cmap(cmap, len(uniq))
        color_map = {u: palette(i) for i, u in enumerate(uniq)}
        colors = [color_map[str(v)] for v in values]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.6, linewidths=0)
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=color_map[u], markersize=7, label=u)
                   for u in uniq]
        ax.legend(handles=handles, title=cbar_label, fontsize=8,
                  loc="upper right", framealpha=0.7)
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=4, alpha=0.6,
                        linewidths=0, cmap="viridis")
        plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out_path}")


def plot_umaps(emb, meta, out_dir: Path):
    if "condition" in meta.columns:
        _scatter_png(emb, meta["condition"], "UMAP — condition",
                     "condition", out_dir / "umap_condition.png",
                     cmap="tab10", categorical=True)
    if "replicate" in meta.columns:
        _scatter_png(emb, meta["replicate"], "UMAP — replicate",
                     "replicate", out_dir / "umap_replicate.png",
                     cmap="Set1", categorical=True)
    _scatter_png(emb, meta["recon_err"], "UMAP — reconstruction error",
                 "MSE", out_dir / "umap_recon_err.png",
                 cmap="viridis", categorical=False)


# ── Step 3: traversals ────────────────────────────────────────────────────────

@torch.no_grad()
def _decode(model, z: torch.Tensor) -> np.ndarray:
    with _silence():
        recon = model.vae.decoder(z)
    return recon[:, :model.nc_img].cpu().numpy()


def plot_traversals(model, Z, device, out_dir: Path, steps: int = 7, span: float = 3.0):
    z_dim   = Z.shape[1]
    mu_mean = torch.tensor(Z.mean(0), dtype=torch.float32)
    std     = Z.std(0)
    alphas  = np.linspace(-span, span, steps)

    fig, axes = plt.subplots(z_dim, steps, figsize=(steps * 1.5, z_dim * 1.5))
    axes = np.atleast_2d(axes)

    for d in range(z_dim):
        zs = mu_mean.repeat(steps, 1).clone()
        zs[:, d] = mu_mean[d] + torch.tensor(alphas * std[d], dtype=torch.float32)
        dec = _decode(model, zs.to(device))       # (steps, 2, H, W)
        for s in range(steps):
            ax = axes[d, s]
            ax.imshow(_norm01(dec[s, 0]), cmap="gray")
            ax.axis("off")
            if d == 0:
                ax.set_title(f"{alphas[s]:+.1f}σ", fontsize=8)
        axes[d, 0].set_ylabel(f"z{d}", fontsize=9, rotation=0,
                               labelpad=16, va="center")

    fig.suptitle("Latent traversals — membrane channel", fontsize=11)
    fig.tight_layout()
    out = out_dir / "traversals.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


# ── Step 4: interpolation ────────────────────────────────────────────────────

@torch.no_grad()
def plot_interpolation(model, Z, meta, device, out_dir: Path, steps: int = 9):
    rng = np.random.default_rng(0)
    if "condition" in meta.columns and meta["condition"].nunique() >= 2:
        conds = meta["condition"].astype(str)
        c0, c1 = list(conds.unique())[:2]
        i0 = int(rng.choice(np.where((conds == c0).to_numpy())[0]))
        i1 = int(rng.choice(np.where((conds == c1).to_numpy())[0]))
        tag = f"{c0} → {c1}"
    else:
        i0, i1 = rng.choice(len(Z), size=2, replace=False)
        tag = "cell A → cell B"

    ts   = np.linspace(0, 1, steps)[:, None]
    path = torch.tensor((1 - ts) * Z[i0] + ts * Z[i1], dtype=torch.float32)
    dec  = _decode(model, path.to(device))    # (steps, 2, H, W)

    fig, axes = plt.subplots(2, steps, figsize=(steps * 1.5, 3.2))
    for s in range(steps):
        axes[0, s].imshow(_norm01(dec[s, 0]), cmap="gray"); axes[0, s].axis("off")
        axes[1, s].imshow(_norm01(dec[s, 1]), cmap="gray"); axes[1, s].axis("off")
        axes[0, s].set_title(f"{ts[s,0]:.2f}", fontsize=8)
    axes[0, 0].set_ylabel("membrane", fontsize=9)
    axes[1, 0].set_ylabel("nuclei",   fontsize=9)
    fig.suptitle(f"Latent interpolation: {tag}", fontsize=11)
    fig.tight_layout()
    out = out_dir / "interpolation.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


# ── Step 5: reconstructions ───────────────────────────────────────────────────

@torch.no_grad()
def plot_reconstructions(model, loader, device, out_dir: Path, recon_err, n: int = 6):
    order_best  = np.argsort(recon_err)[:n]
    order_worst = np.argsort(recon_err)[-n:]
    rng         = np.random.default_rng(1)
    order_rand  = rng.choice(len(recon_err), size=n, replace=False)
    wanted = {int(i): None for i in [*order_best, *order_worst, *order_rand]}

    gi = 0
    for batch in loader:
        x_img = batch[model.hparams.image_key]
        mask  = batch[model.hparams.mask_key]
        for j in range(x_img.shape[0]):
            if gi + j in wanted:
                wanted[gi + j] = (x_img[j].clone(), mask[j].clone(),
                                  batch[model.hparams.nuc_mask_key][j].clone())
        gi += x_img.shape[0]
        if all(v is not None for v in wanted.values()):
            break

    groups = [("random", order_rand), ("best recon", order_best), ("worst recon", order_worst)]
    n_rows = len(groups) * 2
    fig, axes = plt.subplots(n_rows, n, figsize=(n * 1.7, n_rows * 1.6))

    for gr, (name, order) in enumerate(groups):
        for k, idx in enumerate(order):
            entry = wanted.get(int(idx))
            if entry is None:
                continue
            x_img_i, mask_i, nuc_i = entry
            x_in = torch.cat([x_img_i, mask_i.float(), nuc_i.float()],
                              dim=0).unsqueeze(0).to(device)
            with _silence():
                recon, *_ = model.vae(x_in)
            inp = _norm01(x_img_i[0].numpy())
            rec = _norm01(recon[0, 0].cpu().numpy())
            axes[gr * 2,     k].imshow(inp, cmap="gray"); axes[gr * 2,     k].axis("off")
            axes[gr * 2 + 1, k].imshow(rec, cmap="gray"); axes[gr * 2 + 1, k].axis("off")
            if k == 0:
                axes[gr * 2,     k].set_ylabel(f"{name}\ninput",  fontsize=8)
                axes[gr * 2 + 1, k].set_ylabel("recon", fontsize=8)
            axes[gr * 2, k].set_title(f"err={recon_err[idx]:.3f}", fontsize=7)

    fig.suptitle("Reconstructions — membrane channel (input / recon)", fontsize=11)
    fig.tight_layout()
    out = out_dir / "reconstructions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


# ── Step 6: latent dim variance bar ───────────────────────────────────────────

def plot_latent_variance(Z, meta, out_dir: Path):
    z_dim = Z.shape[1]
    var   = Z.var(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # bar chart of per-dim variance
    axes[0].bar(range(z_dim), var, color="steelblue")
    axes[0].set_xticks(range(z_dim))
    axes[0].set_xticklabels([f"z{d}" for d in range(z_dim)], fontsize=9)
    axes[0].set_ylabel("variance")
    axes[0].set_title("Active units — latent dim variance")

    # box plots per dim split by condition (if available)
    if "condition" in meta.columns:
        conds = sorted(meta["condition"].astype(str).unique())
        palette = plt.get_cmap("tab10", len(conds))
        for ci, cond in enumerate(conds):
            sel = (meta["condition"].astype(str) == cond).to_numpy()
            positions = np.arange(z_dim) + ci * 0.2 - 0.1 * (len(conds) - 1)
            bp = axes[1].boxplot([Z[sel, d] for d in range(z_dim)],
                                 positions=positions, widths=0.15,
                                 patch_artist=True, showfliers=False,
                                 medianprops=dict(color="k"))
            for patch in bp["boxes"]:
                patch.set_facecolor(palette(ci))
                patch.set_alpha(0.7)
        axes[1].set_xticks(range(z_dim))
        axes[1].set_xticklabels([f"z{d}" for d in range(z_dim)], fontsize=9)
        axes[1].set_title("Latent dims by condition")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=palette(i), alpha=0.7)
                   for i, _ in enumerate(conds)]
        axes[1].legend(handles, conds, title="condition", fontsize=8)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    out = out_dir / "latent_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--zarr",    required=True)
    p.add_argument("--table",   required=True)
    p.add_argument("--out",     default="outputs/latent")
    p.add_argument("--split",   default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device",  default="auto")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--interp-steps", type=int, default=9,
                   help="Number of steps in the interpolation grid")
    p.add_argument("--trav-steps",   type=int, default=7,
                   help="Number of steps per latent dim in traversal grid")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = _pick_device(args.device)

    _section(f"Loading checkpoint  ({device})")
    model = LitVAE.load_from_checkpoint(args.ckpt, map_location=device)
    print(f"    z_dim={model.hparams.z_dim}  nc={model.hparams.nc}  "
          f"beta={model.hparams.beta}")

    _section("Building dataloader")
    dm = MultinucDataModule(
        data_path=args.zarr, cell_table_csv=args.table,
        channels=(0, 1), batch_size=args.batch, num_workers=args.workers,
        augment=False,
    )
    loader = build_loader(dm, args.split, args.batch, args.workers)
    print(f"    split='{args.split}'  cells={len(loader.dataset)}")

    _section("Encoding cells → latent space")
    Z, recon_err, meta = encode_dataset(model, loader, device)
    print(f"    Z shape: {Z.shape}   mean recon_err={recon_err.mean():.4f}")

    meta.to_csv(out_dir / "latents.csv", index=False)
    np.savez(out_dir / "latents.npz", Z=Z, recon_err=recon_err,
             cell_idx=meta.get("cell_idx", pd.Series(range(len(Z)))).to_numpy())
    print(f"    wrote latents.csv + latents.npz")

    _section("UMAP")
    emb = compute_umap(Z, seed=args.seed)
    plot_umaps(emb, meta, out_dir)

    _section("Latent dimension variance")
    plot_latent_variance(Z, meta, out_dir)

    _section("Traversals")
    plot_traversals(model, Z, device, out_dir, steps=args.trav_steps)

    _section("Interpolation")
    plot_interpolation(model, Z, meta, device, out_dir, steps=args.interp_steps)

    _section("Reconstructions")
    plot_reconstructions(model, loader, device, out_dir, recon_err)

    _section("Done")
    print(f"    All PNGs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
