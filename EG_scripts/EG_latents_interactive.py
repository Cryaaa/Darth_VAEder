#%%
from umap import UMAP
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from darth_vaeder.models.lit_vae import LitVAE
from darth_vaeder.datamodules.dataset_EG import BorderCellDataset, BCDataModule, percentile_norm, no_transform
#%%
CHECKPOINT = "/mnt/efs/dl_jrc/student_data/S-EG/project/VAE1/logs/vae/version_38/checkpoints/last.ckpt"
METADATA_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv"

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
# model.eval()


def get_latent_features(model, loader):
    model.eval()
    latents = []
    logvars = []
    recons = []
    metadata_list = []
    source_images = []

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
            source_images.append(x.cpu())

    mus = torch.cat(latents, dim=0)
    logvars = torch.cat(logvars, dim=0)

    recons = torch.cat(recons, dim=0)
    source_images = torch.cat(source_images, dim=0)
    metadata_list = torch.cat(metadata_list, dim=0)
    
    return mus, logvars, recons, metadata_list, source_images

#get all latent features

mus, logvars, recons, metadata_list, source_images = get_latent_features(model, tqdm(test_loader))


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

#%%
"""
Interactive UMAP explorer with image hover.

Usage:
    from umap_explorer import launch_umap_explorer

    launch_umap_explorer(
        df=df,                  # DataFrame with UMAP cols + metadata
        input_images=inputs,    # (N, C, H, W) or (N, H, W) tensor
        recon_images=recons,    # same shape
        image_ids=ids,          # (N,) tensor/array matching df[id_col]
        umap_x='UMAP1',
        umap_y='UMAP2',
        id_col='cell_id',
        hue_col='condition',    # optional: colour points by this column
        label_cols=['condition', 'plate'],  # optional: show under hover images
    )

    Then open http://localhost:8050 in a browser.
    Requires: pip install dash plotly pillow
"""

import base64
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update
from PIL import Image

# ── image helpers ────────────────────────────────────────────────────────────


def _to_numpy(t):
    """Tensor or array → float32 numpy array."""
    if hasattr(t, "cpu"):
        return t.cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _img_to_b64(img_tensor):
    """Single image tensor (C,H,W) or (H,W) → data-URI PNG string."""
    arr = _to_numpy(img_tensor)
    if arr.ndim == 3:
        # Treat as CHW if first dim is small relative to spatial dims
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = arr.transpose(1, 2, 0)  # CHW → HWC
        # PIL only supports 1, 3, 4 channels; collapse anything else to single channel
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] not in (3, 4):
            arr = arr[:, :, 0]
    # Normalise to [0, 255]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr = (arr * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── main function ─────────────────────────────────────────────────────────────


def launch_umap_explorer(
    df: pd.DataFrame,
    input_images,
    recon_images,
    image_ids,
    umap_x: str = "UMAP1",
    umap_y: str = "UMAP2",
    id_col: str = "image_id",
    hue_col: str | None = None,
    label_cols: list[str] | None = None,
    port: int = 8050,
    img_size: int = 150,
):
    """
    Launch a Dash app at http://localhost:{port}.

    Parameters
    ----------
    df          : DataFrame with UMAP coordinates and metadata.
    input_images: Tensor (N, C, H, W) or (N, H, W) of input images.
    recon_images: Same shape — reconstructions.
    image_ids   : 1-D tensor/array of IDs that match values in df[id_col].
    umap_x/y    : Column names for UMAP coordinates.
    id_col      : Column in df whose values correspond to image_ids.
    hue_col     : Optional column to colour-code points (categorical or numeric).
    label_cols  : Optional list of df columns to display under hover images.
    port        : Port for the local Dash server.
    img_size    : Display size (px) for each hover image.
    """

    # ── Build id → image-index lookup ────────────────────────────────────────
    id_arr = _to_numpy(image_ids).ravel() if hasattr(image_ids, "cpu") else np.asarray(image_ids).ravel()
    # coerce to same type as df[id_col] for matching
    id_to_idx = {v: i for i, v in enumerate(id_arr)}

    # ── Build scatter figure ──────────────────────────────────────────────────
    if hue_col:
        categories = df[hue_col].astype(str).tolist()
        unique_cats = list(dict.fromkeys(categories))
    else:
        categories = ["all"] * len(df)
        unique_cats = ["all"]

    traces = []
    for cat in unique_cats:
        mask = np.array([c == cat for c in categories])
        sub = df[mask]
        traces.append(
            go.Scattergl(
                x=sub[umap_x].values,
                y=sub[umap_y].values,
                mode="markers",
                name=cat if hue_col else "points",
                marker=dict(size=5, opacity=0.7),
                # store the df row id so the callback can look it up
                customdata=sub[id_col].tolist(),
                hovertemplate=f"ID: %{{customdata}}<extra>{cat}</extra>",
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        title="UMAP Explorer — hover a point to see images",
        xaxis_title=umap_x,
        yaxis_title=umap_y,
        hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10),
        height=620,
        legend=dict(itemsizing="constant"),
    )

    # ── Dash layout ───────────────────────────────────────────────────────────
    app = Dash(__name__)

    sidebar_style = {
        "width": "33%",
        "display": "inline-block",
        "vertical-align": "top",
        "padding": "12px",
        "font-family": "monospace",
    }
    img_style = {
        "width": f"{img_size}px",
        "height": f"{img_size}px",
        "image-rendering": "pixelated",
        "border": "1px solid #ccc",
    }

    app.layout = html.Div(
        [
            # ── left: UMAP plot ──────────────────────────────────────────────────
            html.Div(
                dcc.Graph(id="umap-plot", figure=fig, style={"height": "640px"}),
                style={"width": "66%", "display": "inline-block", "vertical-align": "top"},
            ),
            # ── right: image + label panel ───────────────────────────────────────
            html.Div(
                [
                    html.H4("Hover over a point", id="hover-title"),
                    html.Div(id="hover-images"),
                    html.Div(id="hover-labels", style={"margin-top": "8px", "font-size": "12px"}),
                ],
                style=sidebar_style,
            ),
        ]
    )

    # ── Callback ──────────────────────────────────────────────────────────────
    @app.callback(
        Output("hover-title", "children"),
        Output("hover-images", "children"),
        Output("hover-labels", "children"),
        Input("umap-plot", "hoverData"),
    )
    def on_hover(hover_data):
        if hover_data is None:
            return "Hover over a point", no_update, no_update

        point = hover_data["points"][0]
        if "customdata" not in point:
            return no_update, no_update, no_update
        sample_id = point["customdata"]

        # Coerce type for dict lookup (ids may be int/float/str)
        sample_id_key = sample_id
        if sample_id_key not in id_to_idx:
            # try numeric cast
            try:
                sample_id_key = type(next(iter(id_to_idx)))(sample_id)
            except Exception:
                pass

        if sample_id_key not in id_to_idx:
            return f"ID {sample_id} not found", "", ""

        idx = id_to_idx[sample_id_key]

        inp_src = _img_to_b64(input_images[idx])
        rec_src = _img_to_b64(recon_images[idx])

        def _img_block(src, label):
            return html.Div(
                [
                    html.P(label, style={"text-align": "center", "margin": "2px 0", "font-size": "11px"}),
                    html.Img(src=src, style=img_style),
                ],
                style={"display": "inline-block", "margin": "4px"},
            )

        images = html.Div([_img_block(inp_src, "Input"), _img_block(rec_src, "Recon")])

        # Labels from extra columns
        labels = ""
        if label_cols:
            row_mask = df[id_col] == sample_id_key
            if not row_mask.any():
                # try cast
                try:
                    row_mask = df[id_col] == type(df[id_col].iloc[0])(sample_id)
                except Exception:
                    pass
            if row_mask.any():
                row = df[row_mask].iloc[0]
                labels = html.Div(
                    [html.P(f"{col}: {row[col]}", style={"margin": "1px 0"}) for col in label_cols if col in df.columns]
                )

        return f"ID: {sample_id}", images, labels

    print(f"\nUMAP Explorer running → http://localhost:{port}\nCtrl-C to stop.\n")
    app.run(debug=False, port=port)



# %%
launch_umap_explorer(
    df = df_plot,
    input_images = source_images,
    recon_images = recons,
    image_ids = metadata_list.numpy(),
    umap_x = "PCA1",
    umap_y = "PCA2",
    id_col = "image_id",
    hue_col = "migration_stage",
)
# %%
