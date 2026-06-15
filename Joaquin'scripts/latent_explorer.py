"""Latent-space explorer for the multinucleation VAE.

Run this AFTER training to understand what the model learned.  It loads a
checkpoint, encodes every cell into the latent space, and produces a set of
interactive (Plotly HTML) and static (PNG) visualisations.

What you get
------------
1. latents.csv / latents.npz
       The raw per-cell latent vectors (mu), reconstruction error, and all
       metadata.  Everything else is derived from this — keep it for your own
       downstream analysis.

2. embedding_<color>.html        (interactive, ALL cells)
       2-D projection of the z_dim-dimensional latent space (UMAP / PCA / t-SNE)
       coloured by condition, replicate, etc.  Hover shows per-cell metadata.
       --> "Do cells cluster by biological condition?  Does the model just
            memorise which replicate/FOV a cell came from (batch effect)?"

3. embedding_3d.html             (interactive, ALL cells)
       Same projection in 3-D — rotate it to see structure a 2-D squash hides.

4. explorer.html                 (interactive, subsample with THUMBNAILS)
       Same scatter, but hovering a point pops up that cell's actual image in
       the corner.  This is the fun one: fly over the cloud and watch the
       morphology change as you move through latent space.

5. traversals.png                (static)
       Start from the mean cell and walk each latent dimension from -3σ to +3σ,
       decoding at each step.  --> "What morphological factor does each latent
       axis control?"  (size, elongation, nuclei count, intensity, ...)

6. interpolation.png             (static)
       Decode a straight line in latent space between two real cells of
       different conditions.  --> "Is the latent space smooth / meaningful, or
       does it jump?"

7. reconstructions.png           (static)
       Input vs reconstruction for random, best, and worst cells.
       --> "How good is the autoencoder, and where does it fail?"

8. latent_stats.html             (interactive)
       Per-dimension variance ("active units") and violin plots of each latent
       split by condition.  --> "Which latent dims are actually used, and which
       ones separate conditions?"

Usage (server)
--------------
    python "Joaquin'scripts/latent_explorer.py" \
        --ckpt  outputs/checkpoints/best.ckpt \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv \
        --out   outputs/latent \
        --split test \
        --reducer umap

Then open the .html files in any browser (scp them to your laptop, or use an
SSH tunnel).  The PNGs are plain images.

Notes
-----
* Latent = the encoder mean (mu), used deterministically (no sampling), so the
  embedding is reproducible.
* Reconstruction error is the masked MSE on the 2 image channels inside
  pCellmask — the same quantity the model is trained on.
* Optional deps: plotly (required for HTML), umap-learn (best embedding),
  scikit-learn (PCA/t-SNE fallback), pillow (thumbnails).  The script degrades
  gracefully and tells you what to `pip install` if something is missing.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from darth_vaeder.datamodules import MultinucDataModule
from darth_vaeder.datamodules.JS_zarr_datamodule import vae_collate
from darth_vaeder.models import LitVAE


# ══════════════════════════════════════════════════════════════════════════════
# Small utilities
# ══════════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _silence_stdout():
    """Mute the debug print()s inside VAEResNet18.forward during inference."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _norm01(x: np.ndarray) -> np.ndarray:
    """Min-max a single-channel array to [0, 1] for display."""
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo + 1e-6)


def _composite_rgb(patch2: np.ndarray) -> np.ndarray:
    """(2, H, W) membrane+nuclei → (H, W, 3) RGB: red=membrane, green=nuclei."""
    r = _norm01(patch2[0])
    g = _norm01(patch2[1])
    rgb = np.stack([r, g, np.zeros_like(r)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _section(title: str):
    print(f"\n{'─' * 70}\n  {title}\n{'─' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — load model + data, encode every cell
# ══════════════════════════════════════════════════════════════════════════════

def build_loader(dm: MultinucDataModule, split: str, batch_size: int, workers: int):
    """A shuffle-free, drop-nothing loader over the requested split(s)."""
    dm.setup()  # builds train/val/test datasets
    datasets = {
        "train": dm.train_dataset,
        "val":   dm.val_dataset,
        "test":  dm.test_dataset,
    }
    if split == "all":
        ds = ConcatDataset([datasets["train"], datasets["val"], datasets["test"]])
    else:
        ds = datasets[split]
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=vae_collate, drop_last=False,
    )
    return loader


@torch.no_grad()
def encode_dataset(model: LitVAE, loader, device, thumb_idx: set[int],
                   thumb_size: int = 64):
    """Encode all cells → latents, recon error, metadata, and a few thumbnails.

    Returns
    -------
    Z          (N, z_dim)   latent means
    recon_err  (N,)         masked MSE per cell
    meta       DataFrame    one row per cell (from cell_table metadata)
    thumbs     dict[int -> base64 PNG data URI]   for the global rows in thumb_idx
    """
    model.eval().to(device)
    image_key = model.hparams.image_key
    mask_key  = model.hparams.mask_key
    nc_img    = model.nc_img

    Z, errs, meta_rows, thumbs = [], [], [], {}
    gi = 0  # running global cell index (loader is shuffle-free)

    try:
        from PIL import Image
        have_pil = True
    except ImportError:
        have_pil = False

    n_batches = len(loader)
    for bi, batch in enumerate(loader):
        x_img = batch[image_key].to(device)          # (B, 2, H, W)
        mask  = batch[mask_key].to(device)           # (B, 1, H, W)
        x_in  = torch.cat([x_img, mask.float()], dim=1)

        with _silence_stdout():
            recon, _z, mu, _logvar = model.vae(x_in)

        # masked per-cell reconstruction error (matches training loss)
        m2    = (mask > 0).expand_as(x_img).float()
        diff2 = (recon[:, :nc_img] - x_img) ** 2
        err   = (diff2 * m2).sum(dim=[1, 2, 3]) / m2.sum(dim=[1, 2, 3]).clamp_min(1)

        Z.append(mu.cpu().numpy())
        errs.append(err.cpu().numpy())

        # metadata: dict of lists → list of dicts
        md = batch["metadata"]
        bsz = x_img.shape[0]
        for j in range(bsz):
            meta_rows.append({k: md[k][j] for k in md})

        # thumbnails for the chosen subsample
        if have_pil:
            x_cpu = x_img.cpu().numpy()
            for j in range(bsz):
                if gi + j in thumb_idx:
                    rgb = _composite_rgb(x_cpu[j])
                    im  = Image.fromarray(rgb).resize((thumb_size, thumb_size))
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                    thumbs[gi + j] = uri
        gi += bsz

        if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
            print(f"    encoded batch {bi + 1}/{n_batches}  ({gi} cells)")

    Z        = np.concatenate(Z, axis=0)
    recon_err = np.concatenate(errs, axis=0)
    meta = pd.DataFrame(meta_rows)
    meta["recon_err"] = recon_err
    return Z, recon_err, meta, thumbs


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — dimensionality reduction
# ══════════════════════════════════════════════════════════════════════════════

def reduce_dims(Z: np.ndarray, method: str, n_components: int, seed: int = 42):
    """Project latents to n_components.  Falls back gracefully if a lib is missing."""
    method = method.lower()
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=seed)
            return reducer.fit_transform(Z), "UMAP"
        except ImportError:
            print("    [warn] umap-learn not installed → falling back to PCA. "
                  "`pip install umap-learn` for nicer embeddings.")
            method = "pca"

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            return TSNE(n_components=n_components, random_state=seed,
                        init="pca").fit_transform(Z), "t-SNE"
        except ImportError:
            print("    [warn] scikit-learn not installed → falling back to PCA.")
            method = "pca"

    # PCA (also the universal fallback)
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=seed).fit_transform(Z), "PCA"
    except ImportError:
        # last-resort manual PCA via SVD — no sklearn needed
        Zc = Z - Z.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        return (U[:, :n_components] * S[:n_components]), "PCA (numpy SVD)"


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — interactive Plotly embeddings
# ══════════════════════════════════════════════════════════════════════════════

def _require_plotly():
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        print("    [warn] plotly not installed → skipping interactive HTML. "
              "`pip install plotly` to enable.")
        return False


def plot_embedding_2d(emb, meta, color_cols, label, out_dir: Path):
    if not _require_plotly():
        return
    import plotly.express as px

    hover = [c for c in ("condition", "replicate", "image_name", "cell_idx",
                         "recon_err") if c in meta.columns]
    for color in color_cols:
        if color not in meta.columns:
            continue
        df = meta.copy()
        df["x"], df["y"] = emb[:, 0], emb[:, 1]
        fig = px.scatter(
            df, x="x", y="y", color=df[color].astype(str),
            hover_data=hover, opacity=0.7,
            title=f"Latent space ({label}) — coloured by {color}",
            labels={"color": color},
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(legend_title_text=color, width=950, height=750)
        out = out_dir / f"embedding_{color}.html"
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"    wrote {out}")


def plot_embedding_3d(emb3, meta, color, label, out_dir: Path):
    if not _require_plotly() or color not in meta.columns:
        return
    import plotly.express as px
    df = meta.copy()
    df["x"], df["y"], df["z"] = emb3[:, 0], emb3[:, 1], emb3[:, 2]
    hover = [c for c in ("condition", "replicate", "image_name", "recon_err")
             if c in meta.columns]
    fig = px.scatter_3d(df, x="x", y="y", z="z", color=df[color].astype(str),
                        hover_data=hover, opacity=0.7,
                        title=f"Latent space 3-D ({label}) — coloured by {color}")
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(width=950, height=800)
    out = out_dir / "embedding_3d.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"    wrote {out}")


def plot_explorer_with_thumbnails(emb, meta, thumbs, color, label, out_dir: Path):
    """Scatter where hovering a point shows that cell's image in the corner."""
    if not _require_plotly():
        return
    if not thumbs:
        print("    [warn] no thumbnails (pillow missing?) → skipping explorer.html")
        return
    import plotly.graph_objects as go

    idx = np.array(sorted(thumbs.keys()))
    sub = meta.iloc[idx].reset_index(drop=True)
    xy  = emb[idx]
    uris = [thumbs[i] for i in idx]

    color_vals = sub[color].astype(str) if color in sub.columns else None
    groups = color_vals.unique() if color_vals is not None else ["all"]

    fig = go.Figure()
    for g in groups:
        if color_vals is not None:
            sel = (color_vals == g).to_numpy()
        else:
            sel = np.ones(len(sub), dtype=bool)
        cd = np.column_stack([
            np.array(uris, dtype=object)[sel],
            sub.get("condition", pd.Series([""] * len(sub))).to_numpy()[sel],
            sub.get("replicate", pd.Series([""] * len(sub))).to_numpy()[sel],
            sub.get("cell_idx",  pd.Series([""] * len(sub))).to_numpy()[sel],
            np.round(sub["recon_err"].to_numpy()[sel], 4),
        ])
        fig.add_trace(go.Scatter(
            x=xy[sel, 0], y=xy[sel, 1], mode="markers", name=str(g),
            marker=dict(size=6, opacity=0.75), customdata=cd,
            hovertemplate=("cond=%{customdata[1]}<br>rep=%{customdata[2]}"
                           "<br>idx=%{customdata[3]}<br>err=%{customdata[4]}"
                           "<extra></extra>"),
        ))
    fig.update_layout(
        title=f"Latent explorer ({label}) — hover a point to see the cell",
        width=950, height=750, legend_title_text=color,
    )

    # JS injected into the static HTML: pop the hovered cell's image in a corner.
    post = """
    var gd = document.getElementById('{plot_id}');
    var img = document.createElement('img');
    img.style = 'position:fixed;top:12px;right:12px;width:180px;height:180px;'
              + 'border:2px solid #444;background:#000;z-index:1000;'
              + 'image-rendering:pixelated;border-radius:6px;';
    document.body.appendChild(img);
    var cap = document.createElement('div');
    cap.style = 'position:fixed;top:196px;right:12px;width:184px;color:#ccc;'
              + 'font:11px monospace;text-align:center;z-index:1000;';
    cap.innerText = 'hover a point';
    document.body.appendChild(cap);
    gd.on('plotly_hover', function(d){
        var cd = d.points[0].customdata;
        if (cd && cd[0]) { img.src = cd[0]; cap.innerText =
            'cond=' + cd[1] + '  rep=' + cd[2]; }
    });
    """
    out = out_dir / "explorer.html"
    fig.write_html(out, include_plotlyjs="cdn", post_script=post)
    print(f"    wrote {out}  (interactive thumbnails)")


def plot_latent_stats(Z, meta, out_dir: Path):
    """Active-units bar + per-dim violins split by condition."""
    if not _require_plotly():
        return
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    z_dim = Z.shape[1]
    var = Z.var(axis=0)
    order = np.argsort(var)[::-1]

    has_cond = "condition" in meta.columns
    n_top = min(6, z_dim)
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.35, 0.65],
        subplot_titles=("Latent dimension variance (active units)",
                        "Distribution of top latent dims" +
                        (" by condition" if has_cond else "")),
    )
    fig.add_trace(go.Bar(x=[f"z{d}" for d in range(z_dim)], y=var,
                         marker_color="steelblue", name="variance"),
                  row=1, col=1)

    for d in order[:n_top]:
        if has_cond:
            for cond in meta["condition"].astype(str).unique():
                vals = Z[(meta["condition"].astype(str) == cond).to_numpy(), d]
                fig.add_trace(go.Violin(y=vals, name=f"z{d}|{cond}",
                                        legendgroup=cond, scalegroup=f"z{d}",
                                        box_visible=True, meanline_visible=True),
                              row=2, col=1)
        else:
            fig.add_trace(go.Violin(y=Z[:, d], name=f"z{d}", box_visible=True),
                          row=2, col=1)

    fig.update_layout(height=900, width=950, showlegend=True,
                      title_text="Latent dimension statistics")
    out = out_dir / "latent_stats.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"    wrote {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — decoder-based visualisations (traversals, interpolation, recon)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _decode(model, z: torch.Tensor) -> np.ndarray:
    """z (B, z_dim) → (B, 2, H, W) numpy (image channels only)."""
    with _silence_stdout():
        recon = model.vae.decoder(z)
    return recon[:, :model.nc_img].cpu().numpy()


def plot_traversals(model, Z, device, out_dir: Path, steps: int = 7, span: float = 3.0):
    """Walk each latent dim from -span·σ to +span·σ around the mean cell."""
    z_dim = Z.shape[1]
    mu_mean = torch.tensor(Z.mean(0), dtype=torch.float32)
    std = Z.std(0)
    alphas = np.linspace(-span, span, steps)

    fig, axes = plt.subplots(z_dim, steps, figsize=(steps * 1.6, z_dim * 1.6))
    axes = np.atleast_2d(axes)
    for d in range(z_dim):
        zs = mu_mean.repeat(steps, 1).clone()
        zs[:, d] = mu_mean[d] + torch.tensor(alphas * std[d], dtype=torch.float32)
        dec = _decode(model, zs.to(device))            # (steps, 2, H, W)
        for s in range(steps):
            ax = axes[d, s]
            ax.imshow(_norm01(dec[s, 0]), cmap="gray")   # membrane channel
            ax.axis("off")
            if d == 0:
                ax.set_title(f"{alphas[s]:+.1f}σ", fontsize=8)
        axes[d, 0].set_ylabel(f"z{d}", fontsize=9, rotation=0, labelpad=14,
                              va="center")
    fig.suptitle("Latent traversals (membrane channel) — each row is one latent dim",
                 y=1.005)
    fig.tight_layout()
    out = out_dir / "traversals.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


def plot_interpolation(model, Z, meta, device, out_dir: Path, steps: int = 8):
    """Decode a straight line between two real cells (different conditions if possible)."""
    rng = np.random.default_rng(0)
    if "condition" in meta.columns and meta["condition"].nunique() >= 2:
        conds = meta["condition"].astype(str)
        c0, c1 = conds.unique()[:2]
        i0 = rng.choice(np.where((conds == c0).to_numpy())[0])
        i1 = rng.choice(np.where((conds == c1).to_numpy())[0])
        tag = f"{c0} → {c1}"
    else:
        i0, i1 = rng.choice(len(Z), size=2, replace=False)
        tag = "cell A → cell B"

    ts = np.linspace(0, 1, steps)[:, None]
    path = torch.tensor((1 - ts) * Z[i0] + ts * Z[i1], dtype=torch.float32)
    dec = _decode(model, path.to(device))

    fig, axes = plt.subplots(2, steps, figsize=(steps * 1.6, 3.4))
    for s in range(steps):
        axes[0, s].imshow(_norm01(dec[s, 0]), cmap="gray"); axes[0, s].axis("off")
        axes[1, s].imshow(_norm01(dec[s, 1]), cmap="gray"); axes[1, s].axis("off")
        axes[0, s].set_title(f"{ts[s,0]:.2f}", fontsize=8)
    axes[0, 0].set_ylabel("membrane", fontsize=9)
    axes[1, 0].set_ylabel("nuclei", fontsize=9)
    fig.suptitle(f"Latent interpolation: {tag}", y=1.03)
    fig.tight_layout()
    out = out_dir / "interpolation.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


@torch.no_grad()
def plot_reconstructions(model, loader, device, out_dir: Path, recon_err, n=6):
    """Input vs reconstruction for random / best / worst cells (by recon error)."""
    image_key = model.hparams.image_key
    mask_key  = model.hparams.mask_key

    order_best  = np.argsort(recon_err)[:n]
    order_worst = np.argsort(recon_err)[-n:]
    rng = np.random.default_rng(1)
    order_rand  = rng.choice(len(recon_err), size=n, replace=False)
    wanted = {int(i): None for i in [*order_best, *order_worst, *order_rand]}

    # second shuffle-free pass to grab exactly the cells we need
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

    groups = [("random", order_rand), ("best", order_best), ("worst", order_worst)]
    fig, axes = plt.subplots(len(groups) * 2, n, figsize=(n * 1.7, len(groups) * 3.4))
    for gr, (name, order) in enumerate(groups):
        for k, idx in enumerate(order):
            x_img, mask = wanted[int(idx)]
            x_in = torch.cat([x_img, mask.float()], dim=0).unsqueeze(0).to(device)
            with _silence_stdout():
                recon, *_ = model.vae(x_in)
            inp = _norm01(x_img[0].numpy())
            rec = _norm01(recon[0, 0].cpu().numpy())
            axes[gr * 2,     k].imshow(inp, cmap="gray"); axes[gr * 2,     k].axis("off")
            axes[gr * 2 + 1, k].imshow(rec, cmap="gray"); axes[gr * 2 + 1, k].axis("off")
            if k == 0:
                axes[gr * 2,     k].set_ylabel(f"{name}\ninput", fontsize=8)
                axes[gr * 2 + 1, k].set_ylabel("recon", fontsize=8)
            axes[gr * 2, k].set_title(f"err={recon_err[idx]:.3f}", fontsize=7)
    fig.suptitle("Reconstructions (membrane channel): input vs decoded", y=1.005)
    fig.tight_layout()
    out = out_dir / "reconstructions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",  required=True, help="Path to LitVAE checkpoint (.ckpt)")
    p.add_argument("--zarr",  required=True, help="Path to multinucleation.zarr")
    p.add_argument("--table", required=True, help="Path to cell_table.csv")
    p.add_argument("--out",   default="outputs/latent", help="Output directory")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--reducer", default="umap", choices=["umap", "tsne", "pca"])
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device",  default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--max-thumbs", type=int, default=1500,
                   help="Max cells to embed thumbnails for in explorer.html")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.device)

    _section(f"Loading checkpoint  ({device})")
    model = LitVAE.load_from_checkpoint(args.ckpt, map_location=device)
    print(f"    z_dim={model.hparams.z_dim}  nc={model.hparams.nc}  "
          f"beta={model.hparams.beta}  image_key={model.hparams.image_key}")

    _section("Building dataloader")
    dm = MultinucDataModule(
        data_path=args.zarr, cell_table_csv=args.table,
        channels=(0, 1), batch_size=args.batch, num_workers=args.workers,
        augment=False,  # NO augmentation — we want the true latent of each cell
    )
    loader = build_loader(dm, args.split, args.batch, args.workers)
    n_total = len(loader.dataset)
    print(f"    split='{args.split}'  cells={n_total}")

    # choose which global cell indices get a thumbnail (uniform subsample)
    rng = np.random.default_rng(args.seed)
    if n_total > args.max_thumbs:
        thumb_idx = set(rng.choice(n_total, size=args.max_thumbs, replace=False).tolist())
    else:
        thumb_idx = set(range(n_total))

    _section("Encoding cells → latent space")
    Z, recon_err, meta, thumbs = encode_dataset(model, loader, device, thumb_idx)
    print(f"    latents: {Z.shape}   mean recon_err={recon_err.mean():.4f}")

    # save raw artefacts first — everything else is derived
    meta.to_csv(out_dir / "latents.csv", index=False)
    np.savez(out_dir / "latents.npz", Z=Z, recon_err=recon_err,
             cell_idx=meta.get("cell_idx", pd.Series(range(len(Z)))).to_numpy())
    print(f"    wrote {out_dir / 'latents.csv'} and latents.npz")

    color_cols = [c for c in ("condition", "replicate") if c in meta.columns]

    _section(f"Dimensionality reduction ({args.reducer})")
    emb2, label = reduce_dims(Z, args.reducer, 2, args.seed)
    plot_embedding_2d(emb2, meta, color_cols, label, out_dir)
    plot_explorer_with_thumbnails(
        emb2, meta, thumbs, color_cols[0] if color_cols else "condition",
        label, out_dir)
    if Z.shape[1] >= 3:
        emb3, _ = reduce_dims(Z, args.reducer, 3, args.seed)
        plot_embedding_3d(emb3, meta, color_cols[0] if color_cols else "condition",
                          label, out_dir)

    _section("Latent dimension statistics")
    plot_latent_stats(Z, meta, out_dir)

    _section("Decoder visualisations")
    plot_traversals(model, Z, device, out_dir)
    plot_interpolation(model, Z, meta, device, out_dir)
    plot_reconstructions(model, loader, device, out_dir, recon_err)

    _section("Done")
    print(f"    All outputs in: {out_dir.resolve()}")
    print("    Open the .html files in a browser; .png files are static images.")


if __name__ == "__main__":
    main()
