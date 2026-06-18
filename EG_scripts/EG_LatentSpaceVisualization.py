#%%import numpy as np
from umap import UMAP
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from itertools import islice
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import inspect
import re
import torch
import torch.nn as nn
import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from captum.attr import visualization as viz
from darth_vaeder.models import LitVAE
from darth_vaeder.datamodules.dataset_EG import BorderCellDataset, BCDataModule, percentile_norm, no_transform
from torch import tensor

#%%
CHECKPOINT = "/mnt/efs/dl_jrc/student_data/S-EG/project/VAE1/logs/vae/version_20/checkpoints/last.ckpt"
METADATA_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv"

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# Import the checkpoint file; load everything automatically (weights, hyperparameters, and settings)
model = LitVAE.load_from_checkpoint(CHECKPOINT, map_location=device, weights_only=False).eval().to(device)
#%%
import zarr
group = zarr.open("/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr")
list(group.group_keys())
#%%
test_ds = BorderCellDataset(
    annotations_file="/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv", 
    zarr_path = "/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr", input_array_name="max_projection", 
    input_mask_name="3D_mask_corrected", normalization_function= percentile_norm,
    )
#%%
test_data_module = BCDataModule(test_ds, batch_size=32, num_workers = 7, spatial_transforms=no_transform, intensity_transforms=no_transform)

test_data_module.setup("predict")
test_loader = test_data_module.predict_dataloader()
# model.eval()


#%%
def get_latent_features(model, loader):
    model.eval()
    latents = []
    logvars = []
    recons = []
    metadata_list = []

    with torch.no_grad():
        for batch in loader:
            x    = batch["source"]
            mask = batch["masks"]
            metadata = batch["metadata_index"]
            x_in  = torch.cat([x, mask], dim=1).to(torch.float).to(device)
            recon, z, mu, logvar = model.vae(x_in)
            latents.append(mu.cpu())
            logvars.append(logvar.cpu())
            recons.append(recon.cpu())
            metadata_list.append(metadata.cpu())

    mus = torch.cat(latents, dim=0)
    logvars = torch.cat(logvars, dim=0)
    recons = torch.cat(recons, dim=0)
    metadata_list = torch.cat(metadata_list, dim=0)
    
    return mus, logvars, recons, metadata_list

#get all latent features

mus, logvars, recons, metadata_list = get_latent_features(model, tqdm(test_loader))


#%%

# make a umap and plot

# make a PCA 

# two columns for umap and pca each and metadata

# extract metadata and plot with plotly/seaborn

# ── numpy array for sklearn/umap ──────────────────────────────────────────────
Z = mus.numpy()  # shape: (N, latent_dim)

# ── 1. PCA (2 components) ─────────────────────────────────────────────────────
print("Running PCA …")
pca = PCA(n_components=2, random_state=42)
Z_pca = pca.fit_transform(Z)
pca_var = pca.explained_variance_ratio_ * 100  # % variance per component

# ── 2. UMAP (2 components) ────────────────────────────────────────────────────
print("Running UMAP …")
reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
Z_umap = reducer.fit_transform(Z)
 
#%%
# ── 3. Load metadata ──────────────────────────────────────────────────────────
meta = pd.read_csv(METADATA_PATH).reset_index(drop=True)


# Sanity check: number of rows should match latent space
assert len(meta) == len(Z), (
    f"Metadata rows ({len(meta)}) ≠ latent vectors ({len(Z)}). "
    "Make sure the CSV and the dataloader iterate in the same order."
)

# Build a combined DataFrame for easy plotting
df_embed = pd.DataFrame({
    "PCA1":  Z_pca[:, 0],
    "PCA2":  Z_pca[:, 1],
    "UMAP1": Z_umap[:, 0],
    "UMAP2": Z_umap[:, 1],
    "image_id": metadata_list.numpy()
})
df_plot = df_embed.merge(meta, on="image_id")
df_plot
#%%

# ── 4. Identify metadata columns to colour by ─────────────────────────────────
# Exclude any columns that are clearly IDs / paths / free text
SKIP_COLS = {"PCA1", "PCA2", "UMAP1", "UMAP2"}
meta_cols  = [c for c in meta.columns if c not in SKIP_COLS]

# Separate into categorical vs. continuous for colour-map choice
def is_categorical(series, max_unique=20):
    return series.dtype == object or series.nunique() <= max_unique

cat_cols  = [c for c in meta_cols if     is_categorical(df_plot[c])]
cont_cols = [c for c in meta_cols if not is_categorical(df_plot[c])]

# ── 5. Plotting helper ────────────────────────────────────────────────────────
def plot_embedding_grid(df, color_col, pca_var=None):
    """
    Side-by-side PCA | UMAP scatter coloured by `color_col`.
    Works for both categorical and continuous metadata.
    """
    is_cat = is_categorical(df[color_col])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Latent space — coloured by: {color_col}", fontsize=14, fontweight="bold")

    panels = [
        ("PCA1",  "PCA2",  axes[0],
         f"PCA  (PC1 {pca_var[0]:.1f}%  |  PC2 {pca_var[1]:.1f}%)" if pca_var is not None else "PCA"),
        ("UMAP1", "UMAP2", axes[1], "UMAP"),
    ]

    if is_cat:
        palette  = sns.color_palette("tab20", n_colors=df[color_col].nunique())
        hue_order = sorted(df[color_col].dropna().unique().tolist())

        for xc, yc, ax, title in panels:
            sns.scatterplot(
                data=df, x=xc, y=yc, hue=color_col,
                hue_order=hue_order, palette=palette,
                s=18, alpha=0.7, linewidth=0, ax=ax, legend=(ax == axes[-1])
            )
            ax.set_title(title)
            ax.set_xlabel(xc); ax.set_ylabel(yc)

        axes[-1].legend(
            title=color_col, bbox_to_anchor=(1.02, 1),
            loc="upper left", borderaxespad=0, fontsize=8
        )

    else:  # continuous
        vmin, vmax = df[color_col].quantile(0.02), df[color_col].quantile(0.98)
        cmap = "viridis"
        sc = None
        for xc, yc, ax, title in panels:
            sc = ax.scatter(
                df[xc], df[yc], c=df[color_col], cmap=cmap,
                vmin=vmin, vmax=vmax, s=18, alpha=0.7, linewidths=0
            )
            ax.set_title(title)
            ax.set_xlabel(xc); ax.set_ylabel(yc)

        fig.colorbar(sc, ax=axes, label=color_col, shrink=0.8, pad=0.02)

    plt.tight_layout()
    return fig


# ── 6. Generate one figure per metadata column ───────────────────────────────
output_dir = "./latent_space_plots/b1"
import os; os.makedirs(output_dir, exist_ok=True)

all_cols = cat_cols + cont_cols
print(f"\nPlotting {len(all_cols)} metadata columns: {all_cols}\n")

for col in tqdm(all_cols, desc="Generating plots"):
    try:
        fig = plot_embedding_grid(df_plot, color_col=col, pca_var=pca_var)
        safe_name = re.sub(r"[^\w\-]", "_", col)
        fig.savefig(f"{output_dir}/{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ saved: {safe_name}.png")
    except Exception as e:
        print(f"  ✗ skipped '{col}': {e}")

print(f"\nDone. Plots saved to: {output_dir}/")
# %%
