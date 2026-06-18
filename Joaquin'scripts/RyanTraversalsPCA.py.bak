#%%
from json import decoder

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from darth_vaeder.datamodules import MultinucDataModule
from darth_vaeder.JS_models import LitVAE

#%%
# ── config ────────────────────────────────────────────────────────────────────
CKPT      = "/mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/outputs/checkpoints/version_32/last.ckpt"
ZARR      = "/mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr"
TABLE     = "/mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/outputs/cell_table.csv"
OUT_DIR   = Path("outputs/pcaTraversals")       # v32 AE (beta=0)
MAX_CELLS = None         # None = all cells; e.g. 5000 for faster iteration
THUMB_PX  = 80           # pixels per channel in thumbnail (side-by-side → 80×160 total)
img_size  = 96           # matches model: 256 for native, 96 for downsampled
EDGE_THRESHOLD = 5       # drop cropped cells (edge_run_px >= N); must match training run (v31 used 5)
# UMAP
N_NEIGHBORS = 15
MIN_DIST    = 0.1
SEED        = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output  : {OUT_DIR.resolve()}")

#%%
def decode_multi_pc(
    decoder1,
    decoder2,
    Z: np.ndarray,
    img_shape: tuple,
    device: str,
    n_pcs: int = 8,
    n_steps: int = 10,
) -> tuple:
    """
    Traverse each of the top n_pcs principal components from min to max.

    Returns (recons, pca_multi, Z_pca_multi) where recons has shape
    (n_pcs, n_steps, C, H, W).
    """
    sc = StandardScaler().fit(Z)
    Z_scaled = sc.transform(Z)
    pca_multi = PCA(n_components=n_pcs, random_state=42)
    Z_pca_multi = pca_multi.fit_transform(Z_scaled)

    def _inv(z_nd):
        return sc.inverse_transform(pca_multi.inverse_transform(z_nd)).astype(
            np.float32
        )

    z_center = Z_pca_multi.mean(axis=0)
    n_ch, H, W = img_shape
    recons = np.empty((n_pcs, n_steps, n_ch, H, W), dtype=np.float32)

    decoder1.eval()
    decoder2.eval()

    with torch.no_grad():
        for pc_i in range(n_pcs):
            pc_steps = np.linspace(
                Z_pca_multi[:, pc_i].min(), Z_pca_multi[:, pc_i].max(), n_steps
            )
            for step_j, pc_val in enumerate(pc_steps):
                z_pt = z_center.copy()
                z_pt[pc_i] = pc_val
                z_t = torch.tensor(
                    _inv(z_pt[np.newaxis]), dtype=torch.float32, device=device
                )
                out1 = decoder1(z_t).squeeze(0).cpu().numpy()
                out2 = decoder2(z_t).squeeze(0).cpu().numpy()
                recons[pc_i, step_j] = np.concatenate([out1, out2], axis=0)
            if device != "cpu":
                torch.cuda.synchronize()

    print(f"Decoded {n_pcs} PCs × {n_steps} steps")
    return recons, pca_multi, Z_pca_multi

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = LitVAE.load_from_checkpoint(CKPT, map_location=device)
model.eval()
print(f"z_dim={model.hparams.z_dim}  nc={model.hparams.nc}  device={device}")
# %%
from torch.utils.data import ConcatDataset, DataLoader

dm = MultinucDataModule(
    data_path=ZARR,
    cell_table_csv=TABLE,
    channels=(0, 1),
    batch_size=256,
    num_workers=4,
    augment=True,
    pin_memory=False,
    persistent_workers=False,
    img_size=img_size,
    edge_threshold=EDGE_THRESHOLD,
)
dm.setup(None)   # builds train + val + test datasets, edge filter applied once

# all_split_dataset = ConcatDataset([dm.train_dataset, dm.val_dataset, dm.test_dataset])
all_split_dataset = ConcatDataset([dm.test_dataset])

loader = DataLoader(all_split_dataset, batch_size=256, num_workers=4, shuffle=False)

# n_train = len(dm.train_dataset)
# n_val   = len(dm.val_dataset)
n_test  = len(dm.test_dataset)
# print(f"Edge filter kept: train={n_train}  val={n_val}  test={n_test}  total={len(all_split_dataset)}")
print(f"Patch size: {img_size}×{img_size}")
# %%
Zs=np.load("/mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/Joaquin'scripts/outputs/embeddings/vae_ae_v32.npz")['Z']
Zs.shape
decoderCell=model.vae.decoderCell.to(device)
decoderNuc=model.vae.decoderNuc.to(device)
# %%
def DecWrapper(z):
    out1=decoderCell(z)
    out2=decoderNuc(z)
    return torch.cat([out1, out2], dim=1)


# %%
recons, pca_multi, Z_pca_multi=decode_multi_pc(
    decoderCell,
    decoderNuc,
    Z= Zs,
    img_shape= (4,96, 96),
    device=device,
    n_pcs = 8,
    n_steps = 10,
) 
# %%
plt.imshow(recons[0,0,0], cmap='gray')
# %%
recons.shape
# %%

# %%
# ── Cell A: PC1 traversal — membrane (ch0) + nuclei (ch2) ────────────────────
n_steps = recons.shape[1]
pc1_vals = np.linspace(Z_pca_multi[:, 0].min(), Z_pca_multi[:, 0].max(), n_steps)

fig, axes = plt.subplots(2, n_steps, figsize=(n_steps * 1.6, 3.8))
ch_labels = ["membrane (ch0)", "nuclei (ch2)"]
for row, ch in enumerate([0, 2]):
    for col in range(n_steps):
        img = recons[0, col, ch]
        vmin, vmax = img.min(), img.max()
        axes[row, col].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(f"{pc1_vals[col]:.2f}", fontsize=7)
    axes[row, 0].set_ylabel(ch_labels[row], fontsize=9)

fig.suptitle("AE v32 — generation across PC1  (min → max)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "pc1_traversal.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {OUT_DIR / 'pc1_traversal.png'}")

# %%
# ── Cell B: KMeans(3) on PCA space — scatter PC1 vs PC2 ──────────────────────
import pandas as pd
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
labels = kmeans.fit_predict(Z_pca_multi)

ev = pca_multi.explained_variance_ratio_
palette = ["#e6194b", "#4363d8", "#3cb44b"]

fig, ax = plt.subplots(figsize=(7, 5.5))
for k in range(3):
    mask = labels == k
    ax.scatter(
        Z_pca_multi[mask, 0], Z_pca_multi[mask, 1],
        c=palette[k], s=10, alpha=0.5, label=f"cluster {k}  (n={mask.sum()})"
    )
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=120, zorder=5)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)", fontsize=11)
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)", fontsize=11)
ax.set_title("AE v32 latent space — KMeans(3) on PCA", fontsize=12)
ax.legend(fontsize=9, markerscale=2)
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_clusters.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {OUT_DIR / 'pca_clusters.png'}")

# %%
# ── Cell C: condition density per cluster ─────────────────────────────────────
npz_data = np.load(
    "/mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/Joaquin'scripts/outputs/embeddings/vae_ae_v32.npz"
)
cell_idx = npz_data["cell_idx"]
df_table = pd.read_csv(TABLE).set_index("cell_idx")
conditions = df_table.loc[cell_idx, "condition"].values

crosstab = pd.crosstab(labels, conditions)
crosstab.index.name = "cluster"

# per-condition: fraction of each condition's cells that live in each cluster
density = crosstab.div(crosstab.sum(axis=0), axis=1)
print("\n── Density of each condition across clusters (col sums = 1.0) ──")
print(density.round(3).to_string())
print("\n── Cluster composition (row sums = 1.0) ──")
print(crosstab.div(crosstab.sum(axis=1), axis=0).round(3).to_string())

# grouped bar: x=cluster, bars=conditions
x = np.arange(3)
cond_labels = density.columns.tolist()
width = 0.25
cond_colors = {"CTRL": "#4878d0", "MATURE": "#ee854a", "CMs25d": "#6acc65"}

fig, ax = plt.subplots(figsize=(7, 4.5))
for i, cond in enumerate(cond_labels):
    ax.bar(
        x + (i - len(cond_labels) / 2 + 0.5) * width,
        density[cond].values,
        width=width,
        label=cond,
        color=cond_colors.get(cond, f"C{i}"),
    )
ax.set_xticks(x)
ax.set_xticklabels([f"Cluster {k}" for k in range(3)], fontsize=11)
ax.set_ylabel("Fraction of condition in cluster", fontsize=11)
ax.set_title("AE v32 — condition density per PCA cluster", fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_condition_density.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {OUT_DIR / 'cluster_condition_density.png'}")

# %%
# ── Cell D: PCA mosaic — reconstructed thumbnails at PC1/PC2 positions ────────
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

N_MOSAIC = 400   # subsample to avoid total overlap; increase if plot is too sparse
ZOOM     = 0.35  # thumbnail zoom (96px × 0.35 ≈ 34px displayed)

# decode all test-set Zs in one batch
Zs_t = torch.tensor(Zs, dtype=torch.float32, device=device)
with torch.no_grad():
    all_recons_cell = decoderCell(Zs_t).cpu().numpy()   # (N, 2, H, W)

# use membrane channel (ch0) as thumbnail
thumbs = all_recons_cell[:, 0, :, :]  # (N, H, W)

# normalise each thumbnail to [0, 1] for display
t_min = thumbs.min(axis=(1, 2), keepdims=True)
t_max = thumbs.max(axis=(1, 2), keepdims=True)
thumbs = (thumbs - t_min) / (t_max - t_min + 1e-6)

# random subsample
rng = np.random.default_rng(SEED)
idx_sub = rng.choice(len(Zs), size=min(N_MOSAIC, len(Zs)), replace=False)

fig, ax = plt.subplots(figsize=(10, 8))

# background: all points as faint dots colored by cluster
for k in range(3):
    m = labels == k
    ax.scatter(Z_pca_multi[m, 0], Z_pca_multi[m, 1],
               c=palette[k], s=4, alpha=0.15, linewidths=0)

# overlay thumbnails
for i in idx_sub:
    ab = AnnotationBbox(
        OffsetImage(thumbs[i], zoom=ZOOM, cmap="gray"),
        (Z_pca_multi[i, 0], Z_pca_multi[i, 1]),
        frameon=False, pad=0,
    )
    ax.add_artist(ab)

ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)", fontsize=11)
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)", fontsize=11)
ax.set_title(f"AE v32 — reconstruction mosaic on PCA  (n={len(idx_sub)} sampled)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "pca_mosaic.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {OUT_DIR / 'pca_mosaic.png'}")
