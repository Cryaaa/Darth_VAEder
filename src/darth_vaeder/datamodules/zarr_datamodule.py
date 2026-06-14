"""Lightning data module + dataset for the multinucleation.zarr single-cell store.

Store layout (written by migrate_to_zarr.py):

    cell_index/                 global flat table, one row per valid cell
    images/<rep>/<cond>/<img>/  full-resolution FOVs (image, cp_masks, NucleiSeg)
    patches/<rep>/<cond>/<img>/ per-cell 256x256 windows:
        cPatches  (n,256,256,2)  isolated masked cell  [membrane, nuclei]
        bbPatches (n,256,256,2)  context window        [membrane, nuclei]
        cCellmask (n,256,256)    target-cell label mask
        cNucmask  (n,256,256)    nucleus label mask
        bbCellmask / bbNucmask   same masks in bb frame

Each training sample is one cell represented by THREE tensors:

    cPatch    (C, H, W)  float32  isolated cell, raw — normalised by CellPatchNormalize
    bbPatch   (C, H, W)  float32  context window, raw — normalised by ContextPatchNormalize
    cCellmask (1, H, W)  int64    binary mask — not normalised

The VAE encoder receives all three; the decoder reconstructs cPatch only.
Normalisation and augmentation live entirely in transforms.py and are
applied on-the-fly.  Nothing is pre-normalised on disk.

Routing metadata lives in a flat cell_table.csv (one row per cell, built once
by scripts/build_cell_table.py).  The loader carries only integer cell_idx
values; each __getitem__ looks the row up and reads one zarr chunk.

Quick start
-----------
    from darth_vaeder.datamodules import MultinucDataModule

    dm = MultinucDataModule(
        data_path="/scratch/multinucleation.zarr",
        cell_table_csv="outputs/cell_table.csv",
    )
    dm.setup("fit")
    for batch in dm.train_dataloader():
        cp  = batch["cPatch"]     # (B, C, 256, 256) float in [0, 1]
        bb  = batch["bbPatch"]    # (B, C, 256, 256) float in [0, 1]
        msk = batch["cCellmask"]  # (B, 1, 256, 256) int64
        break
"""

from __future__ import annotations

import json
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import zarr
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transforms


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


def load_cell_table(cell_table_csv) -> pd.DataFrame:
    """Read cell_table.csv and index it by cell_idx (column is kept too)."""
    return pd.read_csv(cell_table_csv).set_index("cell_idx", drop=False)


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class CellPatchDataset(Dataset):
    """Map-style dataset: one zarr chunk (one cell) per item.

    Always loads cPatches + bbPatches + cCellmask.  Normalisation and
    augmentation are handled entirely by `transform` (see transforms.py).

    Parameters
    ----------
    data_path   path to multinucleation.zarr
    cell_idx    cell_idx values this split owns (integer list / array)
    table       full cell table from load_cell_table(); shared across splits
    channels    channel indices to load — (0,)=membrane, (1,)=nuclei, (0,1)=both
    transform   sample-dict transform applied after loading (normalise + augment)
    """

    def __init__(
        self,
        data_path,
        cell_idx: Sequence[int],
        table: pd.DataFrame,
        channels: Sequence[int] | None = (0, 1),
        transform: Callable | None = None,
    ):
        self.data_path = str(data_path)
        self.cell_idx  = np.asarray(cell_idx, dtype=np.int64)
        self.table     = table
        self.channels  = list(channels) if channels is not None else None
        self.transform = transform
        self._root     = None   # opened lazily per worker — zarr handles are not fork-safe

    def _ensure_open(self):
        if self._root is None:
            self._root = zarr.open_group(self.data_path, mode="r")

    def __len__(self) -> int:
        return len(self.cell_idx)

    def __getitem__(self, i: int) -> dict:
        self._ensure_open()
        ci  = int(self.cell_idx[i])
        row = self.table.loc[ci]
        rep, cond = str(row["replicate"]), str(row["condition"])
        img, loc  = str(row["image_name"]), int(row["local_cell_index"])
        pg = self._root[f"patches/{rep}/{cond}/{img}"]

        # zarr patches are (n, H, W, C) — permute to (C, H, W) before channel select
        def _load_img(key):
            arr = np.ascontiguousarray(pg[key][loc])        # (H, W, C)
            t   = torch.from_numpy(arr).permute(2, 0, 1).float()  # (C, H, W)
            return t[self.channels] if self.channels is not None else t

        def _load_mask(key):
            arr = np.ascontiguousarray(pg[key][loc])        # (H, W)
            return torch.from_numpy(arr).long().unsqueeze(0)  # (1, H, W)

        sample = {
            "cPatch":    _load_img("cPatches"),
            "bbPatch":   _load_img("bbPatches"),
            "cCellmask": _load_mask("cCellmask"),
            "index":     ci,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample["metadata"] = {k: _py(v) for k, v in row.to_dict().items()}
        return sample


def vae_collate(batch: list[dict]) -> dict:
    """Stack tensors; keep metadata and index without converting strings."""
    out = {}
    tensor_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor)]
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch])
    out["index"] = torch.as_tensor([b["index"] for b in batch], dtype=torch.long)
    meta_keys    = batch[0]["metadata"].keys()
    out["metadata"] = {mk: [b["metadata"][mk] for b in batch] for mk in meta_keys}
    return out


# ════════════════════════════════════════════════════════════════════════════
# Splitting
# ════════════════════════════════════════════════════════════════════════════

def build_splits(
    table: pd.DataFrame,
    ratios=(0.7, 0.15, 0.15),
    split_by: str = "image",
    stratify_by: str | None = "condition",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Partition cell_idx values into train / val / test.

    Parameters
    ----------
    table        full cell table from load_cell_table()
    ratios       (train, val, test) fractions — must sum to 1
    split_by     "image" keeps all cells from one FOV together (recommended —
                 prevents the model from being evaluated on cells whose field
                 it trained on); "cell" shuffles individuals
    stratify_by  column balanced across splits (e.g. "condition"); None skips
    seed         RNG seed for reproducible / shareable splits
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(seed)
    r_tr, r_va, _ = ratios

    if split_by == "cell":
        idx = table["cell_idx"].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(n * r_tr), int(n * r_va)
        return {"train": np.sort(idx[:n_tr]),
                "val":   np.sort(idx[n_tr:n_tr + n_va]),
                "test":  np.sort(idx[n_tr + n_va:])}

    if split_by != "image":
        raise ValueError("split_by must be 'image' or 'cell'")

    imgs   = table.drop_duplicates(["replicate", "image_name"])
    groups = [imgs] if stratify_by is None else [g for _, g in imgs.groupby(stratify_by)]
    chosen = {"train": [], "val": [], "test": []}
    for g in groups:
        keys = g[["replicate", "image_name"]].to_numpy()
        keys = keys[rng.permutation(len(keys))]
        n    = len(keys)
        n_tr, n_va = int(round(n * r_tr)), int(round(n * r_va))
        chosen["train"] += keys[:n_tr].tolist()
        chosen["val"]   += keys[n_tr:n_tr + n_va].tolist()
        chosen["test"]  += keys[n_tr + n_va:].tolist()

    key_series = table["replicate"].astype(str) + "||" + table["image_name"].astype(str)
    out = {}
    for split, keyset in chosen.items():
        wanted = {f"{r}||{i}" for r, i in keyset}
        out[split] = np.sort(table.loc[key_series.isin(wanted), "cell_idx"].to_numpy())
    return out


# ════════════════════════════════════════════════════════════════════════════
# DataModule
# ════════════════════════════════════════════════════════════════════════════

class MultinucDataModule(LightningDataModule):
    """Lightning wrapper around CellPatchDataset.

    Each batch contains three aligned tensors per cell:
        cPatch    (B, C, H, W)  isolated masked cell, normalised by in-mask stats
        bbPatch   (B, C, H, W)  context window, normalised by whole-patch stats
        cCellmask (B, 1, H, W)  binary mask — raw int64, not normalised

    Parameters
    ----------
    data_path           path to multinucleation.zarr
    cell_table_csv      path to per-cell CSV (built by scripts/build_cell_table.py)

    -- what to load --
    channels            channel indices — (0,)=membrane, (1,)=nuclei, (0,1)=both

    -- dataloader --
    batch_size          cells per batch
    num_workers         parallel I/O workers; 0 = main process (easier to debug)
    pin_memory          pin CPU tensors for faster GPU transfer
    persistent_workers  keep workers alive between epochs (recommended: True)
    prefetch_factor     batches prefetched per worker; None = PyTorch default

    -- splitting --
    split_ratios        (train, val, test) fractions, must sum to 1
    split_by            "image" (no FOV leakage, recommended) or "cell"
    stratify_by         table column to balance across splits (e.g. "condition")
    seed                RNG seed for reproducible splits

    -- augmentation / normalisation --
    augment             apply geometry + photometric augmentations to train set;
                        normalisation is always applied regardless of this flag
    cell_norm_low/high  percentile bounds for CellPatchNormalize (cPatch)
    ctx_norm_low/high   percentile bounds for ContextPatchNormalize (bbPatch)
    """

    def __init__(
        self,
        data_path,
        cell_table_csv,
        channels: Sequence[int] | None = (0, 1),
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_by: str = "image",
        stratify_by: str | None = "condition",
        seed: int = 42,
        augment: bool = True,
        cell_norm_low:  float = 1.0,
        cell_norm_high: float = 99.0,
        ctx_norm_low:   float = 1.0,
        ctx_norm_high:  float = 99.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path       = str(data_path)
        self.cell_table_csv  = str(cell_table_csv)
        self.channels        = channels
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.pin_memory      = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.split_ratios    = split_ratios
        self.split_by        = split_by
        self.stratify_by     = stratify_by
        self.seed            = seed
        self.augment         = augment
        self.cell_norm_low   = cell_norm_low
        self.cell_norm_high  = cell_norm_high
        self.ctx_norm_low    = ctx_norm_low
        self.ctx_norm_high   = ctx_norm_high

        # overridable after construction for custom pipelines
        self.train_transform: Callable | None = None
        self.val_transform:   Callable | None = None

        self.table: pd.DataFrame | None = None
        self.splits: dict | None        = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def _make_dataset(self, split_idx, transform) -> CellPatchDataset:
        return CellPatchDataset(
            self.data_path, split_idx, self.table,
            channels=self.channels, transform=transform,
        )

    def setup(self, stage: str | None = None):
        if self.table is None:
            self.table = load_cell_table(self.cell_table_csv)
        if self.splits is None:
            self.splits = build_splits(
                self.table, ratios=self.split_ratios, split_by=self.split_by,
                stratify_by=self.stratify_by, seed=self.seed,
            )
        norm_kwargs = dict(
            cell_norm_low=self.cell_norm_low,   cell_norm_high=self.cell_norm_high,
            ctx_norm_low=self.ctx_norm_low,     ctx_norm_high=self.ctx_norm_high,
        )
        train_tf = self.train_transform or build_transforms(self.augment, **norm_kwargs)
        val_tf   = self.val_transform   or build_transforms(False,        **norm_kwargs)

        if stage in ("fit", "validate", None):
            self.train_dataset = self._make_dataset(self.splits["train"], train_tf)
            self.val_dataset   = self._make_dataset(self.splits["val"],   val_tf)
        if stage in ("test", "predict", None):
            self.test_dataset  = self._make_dataset(self.splits["test"],  val_tf)

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

    def train_dataloader(self): return self._loader(self.train_dataset, shuffle=True)
    def val_dataloader(self):   return self._loader(self.val_dataset,   shuffle=False)
    def test_dataloader(self):  return self._loader(self.test_dataset,  shuffle=False)
    def predict_dataloader(self): return self._loader(self.test_dataset, shuffle=False)

    def save_splits(self, path):
        """Write train/val/test cell_idx lists to JSON."""
        if self.splits is None:
            raise RuntimeError("Call setup() first.")
        with open(path, "w") as f:
            json.dump({k: np.asarray(v).tolist() for k, v in self.splits.items()}, f)

    def load_splits(self, path):
        """Load a saved JSON partition before setup() to override auto-splitting."""
        with open(path) as f:
            self.splits = {k: np.asarray(v, dtype=np.int64) for k, v in json.load(f).items()}


# ════════════════════════════════════════════════════════════════════════════
# Optional: dataset-level normalization stats
# ════════════════════════════════════════════════════════════════════════════

def compute_normalization_stats(data_path, table, cell_idx,
                                channels=(0, 1), low=1.0, high=99.0,
                                max_cells=2000, seed=0) -> dict:
    """Estimate global per-channel stats over both patch types for a cell sample.

    Useful for bias analysis or designing a global-normalisation alternative.
    Returns {"cPatch": {...}, "bbPatch": {...}} each with low/high/mean/std lists.
    """
    root = zarr.open_group(str(data_path), mode="r")
    rng  = np.random.default_rng(seed)
    idx  = np.asarray(list(cell_idx), dtype=np.int64)
    if len(idx) > max_cells:
        idx = rng.choice(idx, size=max_cells, replace=False)

    ch  = list(channels)
    acc = {pt: [[] for _ in ch] for pt in ("cPatches", "bbPatches")}
    for ci in idx:
        row = table.loc[int(ci)]
        rep, cond, img, loc = (str(row["replicate"]), str(row["condition"]),
                               str(row["image_name"]), int(row["local_cell_index"]))
        pg = root[f"patches/{rep}/{cond}/{img}"]
        for pt in acc:
            arr = pg[pt][loc]                           # (H, W, C)
            for j, c in enumerate(ch):
                acc[pt][j].append(arr[..., c].ravel())

    out = {}
    for pt in acc:
        stats = {"low": [], "high": [], "mean": [], "std": []}
        for j in range(len(ch)):
            vals = np.concatenate(acc[pt][j])
            stats["low"].append(float(np.percentile(vals, low)))
            stats["high"].append(float(np.percentile(vals, high)))
            stats["mean"].append(float(vals.mean()))
            stats["std"].append(float(vals.std()))
        out[pt] = stats
    return out
