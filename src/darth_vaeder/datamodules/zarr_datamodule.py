"""Lightning data module + dataset for the multinucleation.zarr single-cell store.

The store has three logical layers (see migrate_to_zarr.py):

    cell_index/                 global flat table, one row per valid cell
        ncells_idx, replicate, condition, image_name,
        local_label_id, local_cell_index
    images/<rep>/<cond>/<img>/  full-resolution FOVs (image, cpmask, nucleiseg)
    patches/<rep>/<cond>/<img>/ per-cell 256x256 windows:
        cPatches  (n,256,256,2)  isolated cell  (membrane, nuclei), masked
        bbPatches (n,256,256,2)  context window (membrane, nuclei), raw
        cCellmask / cNucmask / bbCellmask / bbNucmask  (n,256,256) int labels

One training sample == one cell. The ``cell_index`` IS the routing/metadata layer
that maps a global index to (image group, row within that image), so we never
iterate the hierarchy directly — we shuffle and split global indices, then read
exactly one chunk per cell.

Normalisation is on-the-fly and per-cell: each channel is scaled to ``[0, 1]``
using only the pixels inside the target-cell mask (see ``transforms.py``). The
patches on disk are RAW float32 (uint16-range) — nothing is pre-normalised.

This module only needs numpy, pandas, torch, zarr and lightning, so it can be
imported from a separate training/server repo. Point ``data_path`` at the zarr
store there; nothing here is hard-coded to a local path.

Quick start
-----------
    from darth_vaeder.datamodules import MultinucDataModule

    dm = MultinucDataModule(
        data_path="/scratch/.../multinucleation.zarr",   # <-- you set this
        patch_type="cPatches",            # or "bbPatches"
        batch_size=128,
        split_by="image",                 # no cell leakage across splits
        stratify_by="condition",
        image_metadata_csv="/scratch/.../multinucleation_image_metadata.csv",
    )
    dm.setup("fit")
    for batch in dm.train_dataloader():
        x = batch["image"]                # (B, C, 256, 256) float in [0, 1]
        # VAE: reconstruction target == input (x). batch["metadata"] holds
        # per-cell condition / replicate / pixel size etc. for bias analysis.
        break
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import zarr
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transforms

# Metadata fields always returned per sample (come straight from cell_index).
_CORE_META = ("ncells_idx", "replicate", "condition", "image_name", "local_label_id")
# Microscopy fields joined from the image-metadata CSV (for latent-space bias checks).
_DEFAULT_BIAS_FIELDS = ("microscope", "magnification_x", "pixel_size_um_per_px", "scale_factor")


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class CellPatchDataset(Dataset):
    """One cell per item, read from the zarr store by global index.

    Normalisation lives in ``transform`` (it needs the cell mask), so the mask
    named ``norm_mask_array`` is always loaded into the private ``_normmask`` key,
    passed through the transform pipeline, then dropped before the sample is
    returned.

    Parameters
    ----------
    data_path : str or Path
        Path to multinucleation.zarr.
    frame : pandas.DataFrame
        Rows of the (CSV-enriched) cell index this split owns.
    patch_type : {"cPatches", "bbPatches"}
        Which patch stack to load as the image.
    channels : sequence of int or None
        Channel subset (0=membrane, 1=nuclei). None = all.
    masks : sequence of str
        Mask arrays to also return (e.g. ("cCellmask",) for a masked recon loss).
    transform : callable or None
        Sample-dict transform (normalisation + augmentation). Built by
        ``build_transforms``; if None the image is returned RAW (unnormalised).
    norm_mask_array : str
        Mask array used for per-cell normalisation (default "cCellmask").
    metadata_fields : sequence of str
        Columns of ``frame`` to return under sample["metadata"].
    """

    def __init__(self, data_path, frame: pd.DataFrame, patch_type: str = "cPatches",
                 channels: Sequence[int] | None = (0, 1), masks: Sequence[str] = (),
                 transform: Callable | None = None, norm_mask_array: str = "cCellmask",
                 metadata_fields: Sequence[str] = _CORE_META):
        self.data_path = str(data_path)
        self.patch_type = patch_type
        self.channels = list(channels) if channels is not None else None
        self.masks = tuple(masks)
        self.transform = transform
        self.norm_mask_array = norm_mask_array

        # Pull routing columns into plain arrays for fast, GIL-light __getitem__.
        self._rep = frame["replicate"].to_numpy()
        self._cond = frame["condition"].to_numpy()
        self._img = frame["image_name"].to_numpy()
        self._loc = frame["local_cell_index"].to_numpy()
        self._gidx = frame["ncells_idx"].to_numpy()
        self._meta = {f: frame[f].tolist() for f in metadata_fields if f in frame.columns}

        self._root = None   # opened lazily per worker (fork-safety)

    def _ensure_open(self):
        if self._root is None:
            self._root = zarr.open_group(self.data_path, mode="r")

    def __len__(self) -> int:
        return len(self._gidx)

    def __getitem__(self, i: int) -> dict:
        self._ensure_open()
        rep, cond, img, loc = self._rep[i], self._cond[i], self._img[i], int(self._loc[i])
        pg = self._root[f"patches/{rep}/{cond}/{img}"]

        arr = np.ascontiguousarray(pg[self.patch_type][loc])          # (H, W, C)
        x = torch.from_numpy(arr).permute(2, 0, 1).float()            # (C, H, W)
        if self.channels is not None:
            x = x[self.channels]
        sample = {"image": x, "index": int(self._gidx[i])}

        # Target-cell mask, always loaded for normalisation (private key).
        nm = np.ascontiguousarray(pg[self.norm_mask_array][loc])
        sample["_normmask"] = torch.from_numpy(nm).long().unsqueeze(0)  # (1, H, W)

        for m in self.masks:
            if m == self.norm_mask_array:
                sample[m] = sample["_normmask"].clone()
            else:
                marr = np.ascontiguousarray(pg[m][loc])
                sample[m] = torch.from_numpy(marr).long().unsqueeze(0)

        if self.transform is not None:
            sample = self.transform(sample)
        sample.pop("_normmask", None)

        sample["metadata"] = {f: self._meta[f][i] for f in self._meta}
        return sample


def vae_collate(batch: list[dict]) -> dict:
    """Stack image/mask tensors; gather index and metadata without tensorizing strings."""
    out = {}
    tensor_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor)]
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch])
    out["index"] = torch.as_tensor([b["index"] for b in batch], dtype=torch.long)
    meta_keys = batch[0]["metadata"].keys()
    out["metadata"] = {mk: [b["metadata"][mk] for b in batch] for mk in meta_keys}
    return out


# ════════════════════════════════════════════════════════════════════════════
# Splitting
# ════════════════════════════════════════════════════════════════════════════

def build_splits(index_df: pd.DataFrame, ratios=(0.7, 0.15, 0.15),
                 split_by: str = "image", stratify_by: str | None = "condition",
                 seed: int = 42) -> dict[str, np.ndarray]:
    """Partition global cell indices into train/val/test.

    split_by="image" (recommended) keeps all cells from one FOV in the same split
    so the VAE is never evaluated on cells whose neighbours it trained on.
    split_by="cell" shuffles individual cells (use only if leakage is acceptable).
    stratify_by balances the chosen categorical (e.g. condition) across splits.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(seed)
    r_tr, r_va, _ = ratios

    if split_by == "cell":
        idx = index_df["ncells_idx"].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(n * r_tr), int(n * r_va)
        return {"train": np.sort(idx[:n_tr]),
                "val": np.sort(idx[n_tr:n_tr + n_va]),
                "test": np.sort(idx[n_tr + n_va:])}

    if split_by != "image":
        raise ValueError("split_by must be 'image' or 'cell'")

    # One row per image, optionally grouped by the stratification key.
    imgs = index_df.drop_duplicates(["replicate", "image_name"]).copy()
    train_imgs, val_imgs, test_imgs = [], [], []
    groups = [imgs] if stratify_by is None else [g for _, g in imgs.groupby(stratify_by)]
    for g in groups:
        keys = g[["replicate", "image_name"]].to_numpy()
        keys = keys[rng.permutation(len(keys))]
        n = len(keys)
        n_tr, n_va = int(round(n * r_tr)), int(round(n * r_va))
        train_imgs += keys[:n_tr].tolist()
        val_imgs += keys[n_tr:n_tr + n_va].tolist()
        test_imgs += keys[n_tr + n_va:].tolist()

    def _cells_for(img_keys):
        if not img_keys:
            return np.array([], dtype=np.int64)
        wanted = set(map(tuple, img_keys))
        mask = index_df.apply(lambda r: (r["replicate"], r["image_name"]) in wanted, axis=1)
        return np.sort(index_df.loc[mask, "ncells_idx"].to_numpy())

    return {"train": _cells_for(train_imgs),
            "val": _cells_for(val_imgs),
            "test": _cells_for(test_imgs)}


# ════════════════════════════════════════════════════════════════════════════
# DataModule
# ════════════════════════════════════════════════════════════════════════════

class MultinucDataModule(LightningDataModule):
    """Lightning data module over multinucleation.zarr for VAE training.

    Parameters
    ----------
    data_path : str or Path
        Path to the zarr store (set this in your training/server repo).
    patch_type : {"cPatches", "bbPatches"}
        Isolated cell vs. context window.
    channels : sequence of int or None
        Channel subset (0=membrane, 1=nuclei). None = both.
    masks : sequence of str
        Mask arrays to return alongside the image (e.g. ("cCellmask",)).
    batch_size, num_workers, pin_memory, persistent_workers, prefetch_factor
        Standard DataLoader knobs.
    split_ratios : tuple[float, float, float]
        (train, val, test), must sum to 1.
    split_by : {"image", "cell"}
        Split granularity. "image" prevents cell leakage across splits.
    stratify_by : str or None
        Categorical column to balance across splits (e.g. "condition").
    seed : int
        Split RNG seed (reproducible / shareable splits).
    splits : dict or None
        Precomputed {"train":[...], "val":[...], "test":[...]} global indices.
        If given, overrides split_ratios/split_by (use save_splits/load_splits to
        share an exact partition across runs and collaborators).
    augment : bool
        Apply training augmentations (geometry + photometric). Val/test never
        augment. Ignored if a custom ``train_transform`` is given.
    norm_low, norm_high : float
        Percentiles for the in-mask per-channel normalisation.
    norm_mask_array : str
        Mask array used for normalisation (default "cCellmask").
    train_transform, val_transform : callable or None
        Override the built-in pipelines if you want custom transforms.
    image_metadata_csv : str or None
        Optional microscopy CSV; joined per cell on (sample->replicate,
        stem(filename)->image_name) to attach bias fields to sample["metadata"].
    metadata_fields : sequence of str
        Microscopy columns to attach (in addition to the core cell_index fields).
    """

    def __init__(
        self,
        data_path,
        patch_type: str = "cPatches",
        channels: Sequence[int] | None = (0, 1),
        masks: Sequence[str] = (),
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_by: str = "image",
        stratify_by: str | None = "condition",
        seed: int = 42,
        splits: dict | None = None,
        augment: bool = True,
        norm_low: float = 1.0,
        norm_high: float = 99.0,
        norm_mask_array: str = "cCellmask",
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        image_metadata_csv: str | None = None,
        metadata_fields: Sequence[str] = _DEFAULT_BIAS_FIELDS,
    ):
        super().__init__()
        self.data_path = str(data_path)
        self.patch_type = patch_type
        self.channels = channels
        self.masks = tuple(masks)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.split_ratios = split_ratios
        self.split_by = split_by
        self.stratify_by = stratify_by
        self.seed = seed
        self.splits = splits
        self.augment = augment
        self.norm_low = norm_low
        self.norm_high = norm_high
        self.norm_mask_array = norm_mask_array
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.image_metadata_csv = image_metadata_csv
        self.metadata_fields = tuple(metadata_fields)

        self.index_df: pd.DataFrame | None = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_index(self) -> pd.DataFrame:
        """Read cell_index into a DataFrame and join the microscopy CSV (if given)."""
        root = zarr.open_group(self.data_path, mode="r")
        ci = root["cell_index"]
        index_df = pd.DataFrame({
            "ncells_idx": ci["ncells_idx"][:],
            "replicate": ci["replicate"][:].astype(str),
            "condition": ci["condition"][:].astype(str),
            "image_name": ci["image_name"][:].astype(str),
            "local_cell_index": ci["local_cell_index"][:],
            "local_label_id": ci["local_label_id"][:],
        })
        if self.image_metadata_csv:
            meta = pd.read_csv(self.image_metadata_csv)
            meta = meta.rename(columns={"sample": "replicate"})
            meta["image_name"] = meta["filename"].map(lambda s: Path(str(s)).stem)
            keep = ["replicate", "image_name"] + [c for c in self.metadata_fields if c in meta.columns]
            index_df = index_df.merge(meta[keep].drop_duplicates(["replicate", "image_name"]),
                                      on=["replicate", "image_name"], how="left")
        return index_df

    def _make_dataset(self, split_idx: np.ndarray, transform) -> CellPatchDataset:
        wanted = set(int(x) for x in split_idx)
        frame = self.index_df[self.index_df["ncells_idx"].isin(wanted)].reset_index(drop=True)
        return_meta = tuple(_CORE_META) + tuple(f for f in self.metadata_fields if f in frame.columns)
        return CellPatchDataset(
            self.data_path, frame, patch_type=self.patch_type, channels=self.channels,
            masks=self.masks, transform=transform, norm_mask_array=self.norm_mask_array,
            metadata_fields=return_meta,
        )

    # ── Lightning API ──────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None):
        if self.index_df is None:
            self.index_df = self._load_index()
        if self.splits is None:
            self.splits = build_splits(
                self.index_df, ratios=self.split_ratios, split_by=self.split_by,
                stratify_by=self.stratify_by, seed=self.seed,
            )
        train_tf = self.train_transform or build_transforms(self.augment, self.norm_low, self.norm_high)
        val_tf = self.val_transform or build_transforms(False, self.norm_low, self.norm_high)
        if stage in ("fit", "validate", None):
            self.train_dataset = self._make_dataset(self.splits["train"], train_tf)
            self.val_dataset = self._make_dataset(self.splits["val"], val_tf)
        if stage in ("test", "predict", None):
            self.test_dataset = self._make_dataset(self.splits["test"], val_tf)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            collate_fn=vae_collate,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    # ── reproducible splits ────────────────────────────────────────────────────

    def save_splits(self, path):
        """Persist the exact train/val/test global indices to JSON (shareable)."""
        if self.splits is None:
            raise RuntimeError("Call setup() before save_splits().")
        with open(path, "w") as f:
            json.dump({k: np.asarray(v).tolist() for k, v in self.splits.items()}, f)

    def load_splits(self, path):
        """Load a previously saved split partition (call before setup())."""
        with open(path) as f:
            self.splits = {k: np.asarray(v, dtype=np.int64) for k, v in json.load(f).items()}


# ════════════════════════════════════════════════════════════════════════════
# Optional: dataset-level normalization stats (for analysis / alternative norm)
# ════════════════════════════════════════════════════════════════════════════

def compute_normalization_stats(data_path, indices, patch_type="cPatches",
                                channels=(0, 1), low=1.0, high=99.0,
                                max_cells=2000, seed=0) -> dict:
    """Estimate global per-channel percentile / mean / std over a sample of cells.

    Returns {"low":[...], "high":[...], "mean":[...], "std":[...]}. Not needed for
    the default per-cell masked normalisation, but useful for bias analysis or an
    alternative global-normalisation scheme. Sampling ``max_cells`` keeps it fast.
    """
    root = zarr.open_group(str(data_path), mode="r")
    ci = root["cell_index"]
    rep = ci["replicate"][:].astype(str)
    cond = ci["condition"][:].astype(str)
    img = ci["image_name"][:].astype(str)
    loc = ci["local_cell_index"][:]

    rng = np.random.default_rng(seed)
    idx = np.asarray(list(indices))
    if len(idx) > max_cells:
        idx = rng.choice(idx, size=max_cells, replace=False)

    ch = list(channels)
    acc = [[] for _ in ch]
    for g in idx:
        pg = root[f"patches/{rep[g]}/{cond[g]}/{img[g]}"]
        arr = pg[patch_type][int(loc[g])]          # (H, W, C)
        for j, c in enumerate(ch):
            acc[j].append(arr[..., c].ravel())

    stats = {"low": [], "high": [], "mean": [], "std": []}
    for j in range(len(ch)):
        vals = np.concatenate(acc[j])
        stats["low"].append(float(np.percentile(vals, low)))
        stats["high"].append(float(np.percentile(vals, high)))
        stats["mean"].append(float(vals.mean()))
        stats["std"].append(float(vals.std()))
    return stats
