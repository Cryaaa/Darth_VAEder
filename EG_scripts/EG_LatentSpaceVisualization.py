# Read in Checkpoints, metadata, and an output csv/directory and measure a range of morphology properties, read in the latent space from the checkpoint, and plot Pearson correlation coefficient between the latent space and classical features

#%%
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
from darth_vaeder.models.lit_vae import LitVAE
from darth_vaeder.datamodules.dataset_EG import BorderCellDataset, BCDataModule, percentile_norm, no_transform
from torch import tensor
#%%

CHECKPOINT = "/mnt/efs/dl_jrc/student_data/S-EG/project/VAE1/logs/vae/version_28/checkpoints/last.ckpt"
METADATA_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x_with_morphology.csv"
OUTPUT_CSV_NAME = "data_information_PLC40x_with_LatentsAndMorphology.csv"
output_dir = "./latent_space_plots/AEV28_WithMorphFeatures"

import os; os.makedirs(output_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the checkpoint file; load everything automatically (weights, hyperparameters, and settings)
model = LitVAE.load_from_checkpoint(CHECKPOINT, map_location=device, weights_only=False).eval().to(device)

import zarr
group = zarr.open("/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr")
list(group.group_keys())

test_ds = BorderCellDataset(
    annotations_file="/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv", 
    zarr_path = "/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr", input_array_name="max_projection", 
    input_mask_name="3D_mask_corrected", normalization_function= percentile_norm,
    )

test_data_module = BCDataModule(test_ds, batch_size=32, num_workers = 7, spatial_transforms=no_transform, intensity_transforms=no_transform)

test_data_module.setup("predict")
test_loader = test_data_module.predict_dataloader()

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

#%%
# ── 7. Save image_id, mu_*, logvar_*, PCA, and UMAP alongside the metadata ───
# mus and logvars are tensors of shape (N, latent_dim) — 50 dims each in this case.
# Z_pca / Z_umap were already computed above from the same mus, in the same row order.

n_latent = mus.shape[1]
mu_cols = [f"mu_{i}" for i in range(n_latent)]
logvar_cols = [f"logvar_{i}" for i in range(n_latent)]

df_mu = pd.DataFrame(mus.numpy(), columns=mu_cols)
df_logvar = pd.DataFrame(logvars.numpy(), columns=logvar_cols)
df_dimred = pd.DataFrame({
    "PCA1":  Z_pca[:, 0],
    "PCA2":  Z_pca[:, 1],
    "UMAP1": Z_umap[:, 0],
    "UMAP2": Z_umap[:, 1],
})

df_latents = pd.concat(
    [pd.DataFrame({"image_id": metadata_list.numpy()}), df_mu, df_logvar, df_dimred],
    axis=1,
)

# Re-load the metadata fresh (in case earlier cells mutated `meta` in memory)
meta_full = pd.read_csv(METADATA_PATH).reset_index(drop=True)

# One row per image_id expected on both sides — left merge keeps every metadata row
df_merged = meta_full.merge(df_latents, on="image_id", how="left")

assert df_merged.shape[0] == meta_full.shape[0], (
    "Row count changed after merging latents — check for duplicate or missing image_ids."
)
n_missing = df_merged[mu_cols[0]].isna().sum()
if n_missing:
    print(f"Warning: {n_missing} metadata rows had no matching latent vector (left as NaN).")

df_merged.to_csv(output_dir + "/" + OUTPUT_CSV_NAME, index=False)
print(f"Saved metadata + {n_latent}-D mu/logvar + PCA/UMAP columns to: {output_dir}/{OUTPUT_CSV_NAME}")
print(f"Final shape: {df_merged.shape}")

# %%
# ── 8. Pearson correlation between latent space info and morphology properties
# Correlates mu_*, logvar_*, PCA, and UMAP columns against the original
# morphology / metadata columns (e.g. circularity, convex hull area).
# Result is a (latent_features x morphology_features) matrix, not a square one.
 
EXCLUDE_FROM_CORR = {"image_id", "z-slices", "bit_depth"}
 
morph_cols = [
    c for c in meta_full.columns
    if c not in EXCLUDE_FROM_CORR and pd.api.types.is_numeric_dtype(meta_full[c])
]
latent_cols = mu_cols + logvar_cols + ["PCA1", "PCA2", "UMAP1", "UMAP2"]
 
print(f"Correlating {len(latent_cols)} latent/embedding columns against {len(morph_cols)} morphology columns:")
print(f"  Morphology columns: {morph_cols}")
 
# pandas .corr() only returns a square matrix for one set of columns at a time,
# so compute over the full union and slice out the latent-vs-morphology block.
full_corr = df_merged[latent_cols + morph_cols].corr(method="pearson")
corr_matrix = full_corr.loc[latent_cols, morph_cols]
 
# ── Heatmap ─────────────────────────────────────────────────────────────────
# annot is off by default since this can be 100+ rows; flip to True for a
# small subset if you want to read off exact values.
fig, ax = plt.subplots(figsize=(max(8, len(morph_cols) * 0.6), max(10, len(latent_cols) * 0.25)))
sns.heatmap(
    corr_matrix, annot=False, cmap="coolwarm",
    vmin=-1, vmax=1, ax=ax,
    cbar_kws={"label": "Pearson r"},
)
ax.set_title("Pearson correlation — latent space vs. morphology properties", fontsize=14, fontweight="bold")
ax.set_xlabel("Morphology property")
ax.set_ylabel("Latent dimension (mu / logvar / PCA / UMAP)")
plt.tight_layout()
 
fig.savefig(f"{output_dir}/latent_vs_morphology_pearson_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
 
# Save the full numeric matrix for downstream use
corr_matrix.to_csv(f"{output_dir}/latent_vs_morphology_pearson_correlation.csv")
print(f"Saved correlation heatmap + csv to: {output_dir}/")
 
# ── Ranked summary ─────────────────────────────────────────────────────────
# With 100+ latent dims, the heatmap alone is hard to read — pull out the
# strongest |r| latent/morphology pairs as a quick-reference table.
top_n = 20
corr_long = (
    corr_matrix
    .reset_index()
    .melt(id_vars="index", var_name="morphology_feature", value_name="pearson_r")
    .rename(columns={"index": "latent_feature"})
)
corr_long["abs_r"] = corr_long["pearson_r"].abs()
top_pairs = corr_long.sort_values("abs_r", ascending=False).head(top_n)
 
print(f"\nTop {top_n} strongest latent-feature / morphology correlations:")
print(top_pairs[["latent_feature", "morphology_feature", "pearson_r"]].to_string(index=False))
 
top_pairs.to_csv(f"{output_dir}/top_latent_morphology_correlations.csv", index=False)
print(f"Saved ranked correlation table to: {output_dir}/top_latent_morphology_correlations.csv")





# %%
###############################################################################
# Read in Latent Space from ShapeEmbedLite and plot Pearson Correlation between latent space and classical features

output_dir = "/mnt/efs/dl_jrc/student_data/S-EG/project/ShapeEmbedLite/results_32Points/output_BorderCellDM_ls16_e75_b0.05_lr0.001_idx_loss_rfl_loss_cir_pad"

ShapeEmbedPCA = pd.read_csv("/mnt/efs/dl_jrc/student_data/S-EG/project/ShapeEmbedLite/output_BorderCellDM_ls16_e75_b0.05_lr0.001_idx_loss_rfl_loss_cir_pad_latents_with_labels.csv")


EXCLUDE_FROM_CORR = {"image_id", "z-slices", "bit_depth"}


morph_cols = [
    c for c in meta_full.columns
    if c not in EXCLUDE_FROM_CORR and pd.api.types.is_numeric_dtype(meta_full[c])
]

ShapeEmbed_cols = [
    c for c in ShapeEmbedPCA.columns
    if c not in EXCLUDE_FROM_CORR and pd.api.types.is_numeric_dtype(ShapeEmbedPCA[c])
]
 
print(f"Correlating {len(ShapeEmbed_cols)} latent/embedding columns against {len(morph_cols)} morphology columns:")
print(f"  Morphology columns: {morph_cols}")
 
# pandas .corr() only returns a square matrix for one set of columns at a time,
# so compute over the full union and slice out the latent-vs-morphology block.
full_corr = df_merged[ShapeEmbed_cols + morph_cols].corr(method="pearson")
corr_matrix = full_corr.loc[ShapeEmbed_cols, morph_cols]

# ── Heatmap ─────────────────────────────────────────────────────────────────
# annot is off by default since this can be 100+ rows; flip to True for a
# small subset if you want to read off exact values.
fig, ax = plt.subplots(figsize=(max(8, len(morph_cols) * 0.6), max(10, len(ShapeEmbed_cols) * 0.25)))
sns.heatmap(
    corr_matrix, annot=False, cmap="coolwarm",
    vmin=-1, vmax=1, ax=ax,
    cbar_kws={"label": "Pearson r"},
)
ax.set_title("Pearson correlation — latent space vs. morphology properties", fontsize=14, fontweight="bold")
ax.set_xlabel("Morphology property")
ax.set_ylabel("Latent dimension (mu / logvar / PCA / UMAP)")
plt.tight_layout()
 
fig.savefig(f"{output_dir}/latent_vs_morphology_pearson_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
 
# Save the full numeric matrix for downstream use
corr_matrix.to_csv(f"{output_dir}/latent_vs_morphology_pearson_correlation.csv")
print(f"Saved correlation heatmap + csv to: {output_dir}/")
 
# ── Ranked summary ─────────────────────────────────────────────────────────
# With 100+ latent dims, the heatmap alone is hard to read — pull out the
# strongest |r| latent/morphology pairs as a quick-reference table.
top_n = 20
corr_long = (
    corr_matrix
    .reset_index()
    .melt(id_vars="index", var_name="morphology_feature", value_name="pearson_r")
    .rename(columns={"index": "latent_feature"})
)
corr_long["abs_r"] = corr_long["pearson_r"].abs()
top_pairs = corr_long.sort_values("abs_r", ascending=False).head(top_n)
 
print(f"\nTop {top_n} strongest latent-feature / morphology correlations:")
print(top_pairs[["latent_feature", "morphology_feature", "pearson_r"]].to_string(index=False))
 
top_pairs.to_csv(f"{output_dir}/top_latent_morphology_correlations.csv", index=False)
print(f"Saved ranked correlation table to: {output_dir}/top_latent_morphology_correlations.csv")
# %%
