# Read in Checkpoints, metadata, and an output csv/directory and measure a range of morphology properties, read in the latent space from the checkpoint, and plot Pearson correlation coefficient between the latent space and classical features

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────
METADATA_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x_with_morphology.csv"

output_dir = "/mnt/efs/dl_jrc/student_data/S-EG/project/ShapeEmbedLite/results_32Points/output_BorderCellDM_ls16_e75_b0.05_lr0.001_idx_loss_rfl_loss_cir_pad"

ShapeEmbedPCA = pd.read_csv(
    "/mnt/efs/dl_jrc/student_data/S-EG/project/ShapeEmbedLite/"
    "output_BorderCellDM_ls16_e75_b0.05_lr0.001_idx_loss_rfl_loss_cir_pad_latents_with_labels.csv"
)

meta_full = pd.read_csv(METADATA_PATH)

# ── Merge on image identifier ───────────────────────────────────────────────
# meta_full uses "image_ID", ShapeEmbedPCA uses "name" for the same image label.
print(f"meta_full: {meta_full.shape[0]} rows, {meta_full.shape[1]} cols")
print(f"ShapeEmbedPCA: {ShapeEmbedPCA.shape[0]} rows, {ShapeEmbedPCA.shape[1]} cols")

df_merged = pd.merge(
    meta_full,
    ShapeEmbedPCA,
    left_on="image_id",
    right_on="name",
    how="inner",
    suffixes=("_meta", "_shapeembed"),
)
print(f"df_merged after inner join on image_ID/name: {df_merged.shape[0]} rows")

if df_merged.shape[0] == 0:
    raise ValueError(
        "No rows matched between meta_full['image_ID'] and ShapeEmbedPCA['name']. "
        "Check that the two ID columns actually share the same label format "
        "(e.g. same casing, same file extension presence/absence)."
    )

# ── Column selection ────────────────────────────────────────────────────────
EXCLUDE_FROM_CORR = {"image_id", "z-slices", "bit_depth", "name"}

morph_cols = [
    c for c in meta_full.columns
    if c.lower() not in EXCLUDE_FROM_CORR and pd.api.types.is_numeric_dtype(meta_full[c])
]

ShapeEmbed_cols = [
    c for c in ShapeEmbedPCA.columns
    if c.lower() not in EXCLUDE_FROM_CORR and pd.api.types.is_numeric_dtype(ShapeEmbedPCA[c])
]

# If pandas added suffixes due to overlapping column names on merge, make sure
# we're referencing the post-merge column names that actually exist in df_merged.
morph_cols = [c if c in df_merged.columns else f"{c}_meta" for c in morph_cols]
ShapeEmbed_cols = [c if c in df_merged.columns else f"{c}_shapeembed" for c in ShapeEmbed_cols]

print(f"Correlating {len(ShapeEmbed_cols)} ShapeEmbed/PCA columns against {len(morph_cols)} morphology columns:")
print(f"  Morphology columns: {morph_cols}")
print(f"  ShapeEmbed columns: {ShapeEmbed_cols}")

# pandas .corr() only returns a square matrix for one set of columns at a time,
# so compute over the full union and slice out the ShapeEmbed-vs-morphology block.
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
ax.set_title("Pearson correlation — ShapeEmbed/PCA vs. morphology properties", fontsize=14, fontweight="bold")
ax.set_xlabel("Morphology property")
ax.set_ylabel("ShapeEmbed dimension (latent / PCA)")
plt.tight_layout()

fig.savefig(f"{output_dir}/shapeembed_vs_morphology_pearson_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# Save the full numeric matrix for downstream use
corr_matrix.to_csv(f"{output_dir}/shapeembed_vs_morphology_pearson_correlation.csv")
print(f"Saved correlation heatmap + csv to: {output_dir}/")

# ── Ranked summary ─────────────────────────────────────────────────────────
# With many ShapeEmbed dims, the heatmap alone is hard to read — pull out the
# strongest |r| ShapeEmbed/morphology pairs as a quick-reference table.
top_n = 20
corr_long = (
    corr_matrix
    .reset_index()
    .melt(id_vars="index", var_name="morphology_feature", value_name="pearson_r")
    .rename(columns={"index": "shapeembed_feature"})
)
corr_long["abs_r"] = corr_long["pearson_r"].abs()
top_pairs = corr_long.sort_values("abs_r", ascending=False).head(top_n)

print(f"\nTop {top_n} strongest ShapeEmbed-feature / morphology correlations:")
print(top_pairs[["shapeembed_feature", "morphology_feature", "pearson_r"]].to_string(index=False))

top_pairs.to_csv(f"{output_dir}/top_shapeembed_morphology_correlations.csv", index=False)
print(f"Saved ranked correlation table to: {output_dir}/top_shapeembed_morphology_correlations.csv")
# %%
