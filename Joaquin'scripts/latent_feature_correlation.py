"""Pearson / Spearman correlation: VAE v31 & AE v32 latent spaces vs classical features.

Re-encodes the test split (1,517 cells, seed=42, deterministic) through both
checkpoints, computes PCA-2 and UMAP-2 on each model's mu, joins with
classical morphology features from cell_table.csv, and saves three figures:

  1. features (x) × AE-v32 latent (y)  — mu, logvar, PCA, UMAP
  2. features (x) × VAE-v31 latent (y) — mu, logvar, PCA, UMAP
  3. VAE-v31 latent (x) × AE-v32 latent (y) — direct cross-model comparison

Each output dir gets: heatmap PNG, Spearman heatmap PNG, full CSV, top-20 CSV.
Compact PCA/UMAP-only heatmaps are also saved for quick reading.

Usage
-----
    python "Joaquin'scripts/latent_feature_correlation.py" \\
        --zarr     /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \\
        --table    outputs/cell_table.csv \\
        --vae-ckpt outputs/checkpoints/version_31/last.ckpt \\
        --ae-ckpt  outputs/checkpoints/version_32/last.ckpt
"""

import argparse, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as SkPCA
import umap as umap_lib
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from darth_vaeder.JS_models import LitVAE
from darth_vaeder.datamodules import MultinucDataModule
from _feature_explorer_common import load_and_filter

_SCRIPT_DIR  = Path(__file__).parent
_DEFAULT_OUT = str(_SCRIPT_DIR / "outputs" / "correlation")


# ─── encoding ────────────────────────────────────────────────────────────────

def encode_model(ckpt_path: str, dm, device, table_path: str, tag: str):
    """Encode test split → DataFrame keyed by cell_idx.

    Columns: mu_0..z-1, logvar_0..z-1, PCA1, PCA2, UMAP1, UMAP2, cell_idx.
    Returns: df, mu_cols, logvar_cols, pca_var (2-element array %).
    """
    print(f"\n── encoding {tag} ({ckpt_path}) ──")
    model = LitVAE.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    z_dim = model.hparams.z_dim
    print(f"  z_dim={z_dim}  nc={model.hparams.nc}  device={device}")

    loader = DataLoader(dm.test_dataset, batch_size=256, num_workers=4, shuffle=False)

    # reverse-map (rep, cond, img, local_idx) → cell_idx
    t = pd.read_csv(table_path)
    key2ci = {
        (str(r.replicate), str(r.condition), str(r.image_name), int(r.local_cell_index)): int(r.cell_idx)
        for r in t.itertuples()
    }

    all_mu, all_logvar, all_ci = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {tag}"):
            x_img    = batch[model.hparams.image_key].to(device)
            mask     = batch[model.hparams.mask_key].to(device)
            nuc_mask = batch[model.hparams.nuc_mask_key].to(device)
            x_in     = torch.cat([x_img, mask.float(), nuc_mask.float()], dim=1)
            _, _, mu, logvar = model.vae(x_in)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            meta = batch["metadata"]
            for i in range(len(mu)):
                key = (str(meta["replicate"][i]), str(meta["condition"][i]),
                       str(meta["image_name"][i]), int(meta["local_cell_index"][i]))
                all_ci.append(key2ci[key])

    mu_all     = np.concatenate(all_mu,     axis=0)   # (N, z_dim)
    logvar_all = np.concatenate(all_logvar, axis=0)   # (N, z_dim)
    cell_idx   = np.array(all_ci, dtype=np.int64)
    N = len(mu_all)
    print(f"  encoded {N} cells")

    # ── variance / collapse diagnostics ──────────────────────────────────────
    mu_var = mu_all.var(axis=0)
    dead   = int((mu_var < 0.01).sum())
    lv_mean_above_zero = int((logvar_all.mean(axis=0) > -0.1).sum())
    print(f"  mu variance  min={mu_var.min():.4f}  max={mu_var.max():.4f}  "
          f"near-dead dims (var<0.01): {dead}/{z_dim}")
    if tag == "AE":
        print(f"  NOTE: AE (beta=0) logvar is unconstrained — correlations with logvar are uninformative.")
    else:
        print(f"  logvar dims with mean > -0.1 (near-collapsed): {lv_mean_above_zero}/{z_dim}")

    # ── PCA-2 on mu ───────────────────────────────────────────────────────────
    pca     = SkPCA(n_components=2, random_state=42)
    Z_pca   = pca.fit_transform(mu_all)
    pca_var = pca.explained_variance_ratio_ * 100
    print(f"  PCA-2 on mu: PC1={pca_var[0]:.1f}%  PC2={pca_var[1]:.1f}%")

    # ── UMAP-2 on mu ─────────────────────────────────────────────────────────
    print("  running UMAP-2 …")
    xy = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                       random_state=42, verbose=False).fit_transform(mu_all)

    # ── build DataFrame ───────────────────────────────────────────────────────
    mu_cols     = [f"{tag}_mu_{i}"     for i in range(z_dim)]
    logvar_cols = [f"{tag}_logvar_{i}" for i in range(z_dim)]

    df = pd.DataFrame(mu_all,     columns=mu_cols)
    df[logvar_cols]      = logvar_all
    df[f"{tag}_PCA1"]    = Z_pca[:, 0]
    df[f"{tag}_PCA2"]    = Z_pca[:, 1]
    df[f"{tag}_UMAP1"]   = xy[:, 0]
    df[f"{tag}_UMAP2"]   = xy[:, 1]
    df["cell_idx"]       = cell_idx

    return df, mu_cols, logvar_cols, pca_var


# ─── cross-check against saved .npz ─────────────────────────────────────────

def crosscheck_npz(df, mu_cols, npz_path, tag):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        print(f"  [{tag}] cross-check skipped (npz not found: {npz_path})")
        return
    saved = np.load(npz_path)
    s_ci  = saved["cell_idx"]
    d_ci  = df["cell_idx"].values
    overlap = np.intersect1d(s_ci, d_ci)
    print(f"  [{tag}] cross-check vs {npz_path.name}: "
          f"{len(overlap)}/{len(s_ci)} cell_idx match")
    if len(overlap) == len(s_ci) == len(d_ci):
        # align both by cell_idx
        s_order = np.argsort(s_ci)
        d_order = np.argsort(d_ci)
        saved_mu = saved["Z"][s_order]
        enc_mu   = df[mu_cols].values[d_order]
        max_diff = np.abs(saved_mu - enc_mu).max()
        status   = "✓ OK" if max_diff < 1e-3 else "⚠  MISMATCH"
        print(f"  [{tag}] max |mu_saved − mu_reencoded| = {max_diff:.2e}  {status}")
    else:
        print(f"  [{tag}] ⚠  cell_idx sets differ — check split reproducibility")


# ─── heatmap + CSV helpers ────────────────────────────────────────────────────

def _heatmap(corr, title, out_path, method="Pearson"):
    nrows, ncols = corr.shape
    fw = max(8,  ncols * 0.50)
    fh = max(5,  nrows * 0.20)
    fig, ax = plt.subplots(figsize=(fw, fh))
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": f"{method} r"}, linewidths=0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(corr.columns.name or "")
    ax.set_ylabel(corr.index.name   or "")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def save_correlation_block(df, row_cols, col_cols, out_dir, label, row_label="", col_label=""):
    """Compute Pearson + Spearman, save heatmaps, CSVs, and top-20 ranked pairs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df[row_cols + col_cols].dropna()
    if len(sub) < 10:
        print(f"  WARNING: only {len(sub)} rows after dropna — skipping {label}")
        return

    # ── Pearson ───────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_p  = sub.corr(method="pearson")
        full_sp = sub.corr(method="spearman")

    cm_pearson  = full_p.loc[row_cols,  col_cols]
    cm_spearman = full_sp.loc[row_cols, col_cols]
    cm_pearson.index.name  = row_label or "latent"
    cm_pearson.columns.name = col_label or "feature"

    safe = label.replace(" ", "_")
    _heatmap(cm_pearson,  f"{label} — Pearson",   out_dir / f"{safe}_pearson.png",  "Pearson")
    _heatmap(cm_spearman, f"{label} — Spearman",  out_dir / f"{safe}_spearman.png", "Spearman")

    cm_pearson.to_csv( out_dir / f"{safe}_pearson.csv")
    cm_spearman.to_csv(out_dir / f"{safe}_spearman.csv")

    # top-20 Pearson pairs
    corr_long = (cm_pearson.reset_index()
                 .melt(id_vars="index", var_name="col", value_name="pearson_r")
                 .rename(columns={"index": "row"}))
    corr_long["abs_r"] = corr_long["pearson_r"].abs()
    top = corr_long.sort_values("abs_r", ascending=False).head(20)
    top.to_csv(out_dir / f"{safe}_top20_pearson.csv", index=False)
    print(f"  top-5 Pearson |r| pairs:")
    for _, row in top.head(5).iterrows():
        print(f"    {row['row']:40s}  {row['col']:40s}  r={row['pearson_r']:+.3f}")


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",           required=True)
    p.add_argument("--table",          required=True)
    p.add_argument("--vae-ckpt",       required=True,  help="VAE v31 checkpoint")
    p.add_argument("--ae-ckpt",        required=True,  help="AE v32 checkpoint")
    p.add_argument("--out",            default=_DEFAULT_OUT)
    p.add_argument("--img-size",       type=int, default=96)
    p.add_argument("--edge-threshold", type=int, default=5)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output root : {out.resolve()}")
    print(f"Device      : {device}")

    # ── shared datamodule (test split only; same seed as dashboard) ───────────
    print("\n── building datamodule ──")
    dm = MultinucDataModule(
        data_path       = args.zarr,
        cell_table_csv  = args.table,
        channels        = (0, 1),
        batch_size      = 256,
        num_workers     = 4,
        augment         = False,   # no augmentation during encoding
        pin_memory      = False,
        persistent_workers = False,
        img_size        = args.img_size,
        edge_threshold  = args.edge_threshold,
    )
    dm.setup(None)
    n_test = len(dm.test_dataset)
    print(f"  test split : {n_test} cells")

    # ── encode both models ────────────────────────────────────────────────────
    df_vae, mu_cols_vae, lv_cols_vae, pca_var_vae = encode_model(
        args.vae_ckpt, dm, device, args.table, "VAE")
    df_ae,  mu_cols_ae,  lv_cols_ae,  pca_var_ae  = encode_model(
        args.ae_ckpt,  dm, device, args.table, "AE")

    # ── sanity checks ─────────────────────────────────────────────────────────
    print("\n── sanity checks ──")
    ci_vae = set(df_vae["cell_idx"])
    ci_ae  = set(df_ae["cell_idx"])
    assert ci_vae == ci_ae, (
        f"VAE and AE encoded different cell sets! "
        f"VAE only: {len(ci_vae - ci_ae)}  AE only: {len(ci_ae - ci_vae)}"
    )
    assert len(df_vae) == n_test, f"VAE rows {len(df_vae)} ≠ test split {n_test}"
    assert len(df_ae)  == n_test, f"AE rows {len(df_ae)} ≠ test split {n_test}"
    print(f"  ✓ both models encoded identical {n_test} test cells")

    # cross-check mu against saved .npz files
    emb_dir = Path(args.table).parent / "embeddings"
    crosscheck_npz(df_vae, mu_cols_vae, emb_dir / "vae_v31.npz",    "VAE")
    crosscheck_npz(df_ae,  mu_cols_ae,  emb_dir / "vae_ae_v32.npz", "AE")

    # ── classical features ────────────────────────────────────────────────────
    print("\n── loading features ──")
    df_feat, feat_cols = load_and_filter(args.table, args.edge_threshold)

    # ── join: restrict features to test cells ─────────────────────────────────
    print("\n── joining on cell_idx ──")
    df_vae_feat = df_vae.merge(df_feat[["cell_idx"] + feat_cols], on="cell_idx", how="inner")
    df_ae_feat  = df_ae.merge( df_feat[["cell_idx"] + feat_cols], on="cell_idx", how="inner")
    df_cross    = df_vae.merge(df_ae.drop(columns=["cell_idx"]).assign(
                                   **{"cell_idx": df_ae["cell_idx"]}),
                               left_on="cell_idx", right_on="cell_idx", how="inner")

    # The above merge trick doesn't work cleanly — do it properly:
    df_cross = df_vae.merge(df_ae, on="cell_idx", how="inner")

    for name, df in [("VAE+feat", df_vae_feat), ("AE+feat", df_ae_feat), ("AE×VAE", df_cross)]:
        n_nan = df[feat_cols[0] if name != "AE×VAE" else mu_cols_ae[0]].isna().sum()
        print(f"  {name}: {len(df)} rows, NaN in first col: {n_nan}")

    assert len(df_vae_feat) > 0, "VAE-feature join is empty!"
    assert len(df_ae_feat)  > 0, "AE-feature join is empty!"
    assert len(df_cross)    > 0, "AE×VAE join is empty!"
    print(f"  ✓ all joins non-empty")

    # pretty feature names for axis labels
    def feat_short(c):
        return (c.replace("gFeat_cell_", "cell·").replace("gFeat_nuc_", "nuc·")
                 .replace("tFeat_mem_", "mem·").replace("tFeat_nuc_", "nuc·"))
    feat_labels = [feat_short(c) for c in feat_cols]

    # rename feat columns in merged dfs to short names
    rename_map = dict(zip(feat_cols, feat_labels))
    df_vae_feat = df_vae_feat.rename(columns=rename_map)
    df_ae_feat  = df_ae_feat.rename(columns=rename_map)

    # ── latent column lists ───────────────────────────────────────────────────
    vae_compact_cols = ["VAE_PCA1", "VAE_PCA2", "VAE_UMAP1", "VAE_UMAP2"]
    ae_compact_cols  = ["AE_PCA1",  "AE_PCA2",  "AE_UMAP1",  "AE_UMAP2"]
    vae_all_cols     = mu_cols_vae + lv_cols_vae + vae_compact_cols
    ae_all_cols      = mu_cols_ae  + lv_cols_ae  + ae_compact_cols

    pca1_lbl_vae = f"VAE_PCA1 ({pca_var_vae[0]:.1f}%)"
    pca2_lbl_vae = f"VAE_PCA2 ({pca_var_vae[1]:.1f}%)"
    pca1_lbl_ae  = f"AE_PCA1  ({pca_var_ae[0]:.1f}%)"
    pca2_lbl_ae  = f"AE_PCA2  ({pca_var_ae[1]:.1f}%)"

    for df in (df_vae_feat, df_cross):
        df.rename(columns={"VAE_PCA1": pca1_lbl_vae, "VAE_PCA2": pca2_lbl_vae}, inplace=True)
    for df in (df_ae_feat, df_cross):
        df.rename(columns={"AE_PCA1": pca1_lbl_ae, "AE_PCA2": pca2_lbl_ae}, inplace=True)

    vae_compact_cols = [pca1_lbl_vae, pca2_lbl_vae, "VAE_UMAP1", "VAE_UMAP2"]
    ae_compact_cols  = [pca1_lbl_ae,  pca2_lbl_ae,  "AE_UMAP1",  "AE_UMAP2"]
    vae_all_cols = ([c.replace("VAE_PCA1", pca1_lbl_vae).replace("VAE_PCA2", pca2_lbl_vae)
                     for c in vae_all_cols])
    ae_all_cols  = ([c.replace("AE_PCA1",  pca1_lbl_ae). replace("AE_PCA2",  pca2_lbl_ae)
                     for c in ae_all_cols])

    # ── Plot 1: features × VAE ────────────────────────────────────────────────
    print("\n── Plot 1: features × VAE ──")
    vae_out = out / "features_vs_vae"
    save_correlation_block(df_vae_feat, vae_all_cols,     feat_labels, vae_out,
                           "VAE-v31 latent vs features (all dims)",
                           row_label="VAE latent", col_label="feature")
    save_correlation_block(df_vae_feat, vae_compact_cols, feat_labels, vae_out,
                           "VAE-v31 PCA+UMAP vs features (compact)",
                           row_label="VAE embedding", col_label="feature")

    # ── Plot 2: features × AE ─────────────────────────────────────────────────
    print("\n── Plot 2: features × AE ──")
    ae_out = out / "features_vs_ae"
    save_correlation_block(df_ae_feat, ae_all_cols,      feat_labels, ae_out,
                           "AE-v32 latent vs features (all dims)",
                           row_label="AE latent", col_label="feature")
    save_correlation_block(df_ae_feat, ae_compact_cols,  feat_labels, ae_out,
                           "AE-v32 PCA+UMAP vs features (compact)",
                           row_label="AE embedding", col_label="feature")

    # ── Plot 3: AE × VAE ──────────────────────────────────────────────────────
    print("\n── Plot 3: AE latent × VAE latent ──")
    cross_out = out / "ae_vs_vae"
    # mu × mu block (most meaningful)
    save_correlation_block(df_cross, ae_all_cols, vae_all_cols, cross_out,
                           "AE-v32 vs VAE-v31 (all dims)",
                           row_label="AE latent", col_label="VAE latent")
    save_correlation_block(df_cross, ae_compact_cols, vae_compact_cols, cross_out,
                           "AE-v32 vs VAE-v31 (PCA+UMAP compact)",
                           row_label="AE embedding", col_label="VAE embedding")

    print(f"\n✓ Done. All outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
