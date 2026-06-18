"""Shared utilities for feature_umap_explorer.py and feature_pca_explorer.py."""

import io, base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image
from sklearn.preprocessing import QuantileTransformer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_filter(table_path: str, edge_threshold: int = 5):
    """Load CSV, edge-filter, select feature columns, drop NaN rows."""
    df = pd.read_csv(table_path)
    print(f"  loaded       {len(df)} cells")

    df = df[df["edge_run_px"] < edge_threshold].reset_index(drop=True)
    print(f"  edge filter  {len(df)} cells remaining")

    feat_cols = [c for c in df.columns
                 if c.startswith(("gFeat_", "tFeat_")) and "orientation" not in c]
    if not feat_cols:
        raise RuntimeError("No gFeat_/tFeat_ columns — run ComputeFeatures.py first")
    print(f"  features     {len(feat_cols)}")

    n_before = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"  NaN drop     {len(df)} cells ({n_before - len(df)} removed)")
    return df, feat_cols


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_features(X_raw: np.ndarray) -> np.ndarray:
    """QuantileTransformer → Gaussian output.

    Robust to the right-skewed size features (area, perimeter, etc.).
    Each feature is independently mapped to an approximate standard normal,
    so all 33 features contribute comparably to UMAP/PCA distances.
    """
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, X_raw.shape[0]),
        random_state=42,
    )
    return qt.fit_transform(X_raw).astype(np.float32)


# ---------------------------------------------------------------------------
# Condition grouping
# ---------------------------------------------------------------------------

def build_classes(df: pd.DataFrame) -> list:
    """One dict per condition: {name, indices} where indices are df row positions."""
    conditions = list(dict.fromkeys(df["condition"]))
    return [
        {"name": cond, "indices": df.index[df["condition"] == cond].tolist()}
        for cond in conditions
    ]


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

def _normalize_channel(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    denom = float(hi) - float(lo)
    if denom <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / denom, 0.0, 1.0)


def _thumb_b64(mem: np.ndarray, nuc: np.ndarray, size: int) -> str:
    panels = []
    for arr in (mem, nuc):
        pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        pil = pil.resize((size, size), Image.LANCZOS)
        panels.append(np.array(pil))
    composite = np.concatenate(panels, axis=1)
    buf = io.BytesIO()
    Image.fromarray(composite, mode="L").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _process_group(args):
    zarr_path, rep, cond, img, rows, thumb_px = args
    root = zarr.open_group(zarr_path, mode="r")
    pg = root[f"patches/{rep}/{cond}/{img}"]
    cnp = pg["cnPatches"]   # (N, H, W, 2) float32
    results = []
    for _, row in rows.iterrows():
        loc = int(row["local_cell_index"])
        ci  = int(row["cell_idx"])
        mem = _normalize_channel(cnp[loc, :, :, 0], row["norm_mem_lo"], row["norm_mem_hi"])
        nuc = _normalize_channel(cnp[loc, :, :, 1], row["norm_nuc_lo"], row["norm_nuc_hi"])
        results.append((ci, _thumb_b64(mem, nuc, thumb_px)))
    return results


def generate_thumbnails(df: pd.DataFrame, zarr_path: str,
                        thumb_px: int = 48, workers: int = 8) -> dict:
    """Read cnPatches from zarr, return {str(cell_idx): base64_png}."""
    groups: dict = {}
    for _, row in df.iterrows():
        key = (str(row["replicate"]), str(row["condition"]), str(row["image_name"]))
        groups.setdefault(key, []).append(row)

    work = [
        (zarr_path, rep, cond, img, pd.DataFrame(rows), thumb_px)
        for (rep, cond, img), rows in groups.items()
    ]
    print(f"  thumbnails   {len(work)} groups × {workers} workers …")

    thumbnails: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_group, w): w for w in work}
        done = 0
        for fut in as_completed(futs):
            for ci, b64 in fut.result():
                thumbnails[str(ci)] = b64
            done += 1
            if done % 50 == 0 or done == len(work):
                print(f"    {done}/{len(work)} groups", end="\r")
    print(f"\n  generated    {len(thumbnails)} thumbnails")
    return thumbnails


# ---------------------------------------------------------------------------
# Persist normalized features for benchmarking
# ---------------------------------------------------------------------------

def save_normalized_npz(df: pd.DataFrame, X_norm: np.ndarray,
                        feat_cols: list, out_dir: Path) -> None:
    """Save features_normalized.npz keyed by cell_idx (used by benchmark notebook)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "features_normalized.npz"
    np.savez(
        path,
        Z=X_norm.astype(np.float32),
        cell_idx=df["cell_idx"].astype(np.int64).values,
        feat_names=np.array(feat_cols),
    )
    print(f"  saved npz    {path}")
