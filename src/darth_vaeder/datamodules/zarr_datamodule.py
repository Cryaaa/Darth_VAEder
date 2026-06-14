"""Lightning data module + dataset for the multinucleation.zarr single-cell store.

Store layout (written by migrate_to_zarr.py):

    cell_index/                 global flat table, one row per valid cell
    images/<rep>/<cond>/<img>/  full-resolution FOVs (image, cp_masks, NucleiSeg)
    patches/<rep>/<cond>/<img>/ per-cell 256×256 windows:
        cPatches  (n,256,256,2)  isolated cell  [membrane, nuclei], masked
        bbPatches (n,256,256,2)  context window [membrane, nuclei], raw
        cCellmask / cNucmask / bbCellmask / bbNucmask  (n,256,256) int labels

One training sample == one cell.  Cell routing lives in a flat CSV
(outputs/cell_table.csv, built once by scripts/build_cell_table.py).
The data loader carries only integer ``cell_idx`` values; for each it reads
the CSV row → zarr address → one chunk.  Normalisation is on-the-fly and
per-cell (only in-mask pixels, see transforms.MaskedPerChannelNormalize).

Quick start
-----------
    dm = MultinucDataModule(
        data_path="/scratch/multinucleation.zarr",
        cell_table_csv="outputs/cell_table.csv",
    )
    dm.setup("fit")
    for batch in dm.train_dataloader():
        x = batch["image"]   # (B, C, 256, 256) float in [0, 1]
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

    Parameters
    ----------
    data_path       path to multinucleation.zarr
    cell_idx        cell_idx values this split owns (integer list / array)
    table           full cell table from load_cell_table(); shared across splits
    patch_type      "cPatches" (masked, isolated) or "bbPatches" (raw context)
    channels        channel indices to return — (0,)=membrane, (1,)=nuclei,
                    (0,1)=both (default); None keeps all channels as-is
    masks           extra mask arrays to include in the sample dict,
                    e.g. ("cCellmask",) for a masked reconstruction loss
    transform       sample-dict transform (normalisation + augmentations);
                    receives and returns {"image":…, "_normmask":…, …}
    norm_mask_array zarr array used to define "in-mask" pixels for normalisation
    """

    def __init__(
        self,
        data_path,
        cell_idx: Sequence[int],
        table: pd.DataFrame,
        patch_type: str = "cPatches",
        channels: Sequence[int] | None = (0, 1),
        masks: Sequence[str] = (),
        transform: Callable | None = None,
        norm_mask_array: str = "cCellmask",
    ):
        self.data_path = str(data_path)
        self.cell_idx = np.asarray(cell_idx, dtype=np.int64)
        self.table = table
        self.patch_type = patch_type
        self.channels = list(channels) if channels is not None else None
        self.masks = tuple(masks)
        self.transform = transform
        self.norm_mask_array = norm_mask_array
        self._root = None   # opened lazily per worker — zarr handles are not fork-safe

    def _ensure_open(self):
        if self._root is None:
            self._root = zarr.open_group(self.data_path, mode="r")

    def __len__(self) -> int:
        return len(self.cell_idx)

    def __getitem__(self, i: int) -> dict:
        self._ensure_open()
        ci = int(self.cell_idx[i])
        row = self.table.loc[ci]
        rep, cond = str(row["replicate"]), str(row["condition"])
        img, loc = str(row["image_name"]), int(row["local_cell_index"])
        pg = self._root[f"patches/{rep}/{cond}/{img}"]

        # zarr patches are (n, H, W, C) — permute to (C, H, W) before channel select
        arr = np.ascontiguousarray(pg[self.patch_type][loc])
        x = torch.from_numpy(arr).permute(2, 0, 1).float()
        if self.channels is not None:
            x = x[self.channels]
        sample = {"image": x, "index": ci}

        # _normmask drives MaskedPerChannelNormalize; dropped after transform
        nm = np.ascontiguousarray(pg[self.norm_mask_array][loc])
        sample["_normmask"] = torch.from_numpy(nm).long().unsqueeze(0)

        for m in self.masks:
            if m == self.norm_mask_array:
                sample[m] = sample["_normmask"].clone()
            else:
                marr = np.ascontiguousarray(pg[m][loc])
                sample[m] = torch.from_numpy(marr).long().unsqueeze(0)

        if self.transform is not None:
            sample = self.transform(sample)
        sample.pop("_normmask", None)

        sample["metadata"] = {k: _py(v) for k, v in row.to_dict().items()}
        return sample


def vae_collate(batch: list[dict]) -> dict:
    """Stack tensors; keep metadata and index without converting strings."""
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
    split_by     "image" (recommended) keeps all cells from one FOV together,
                 preventing the VAE from being evaluated on cells whose field
                 it was trained on; "cell" shuffles individuals (leaky but
                 maximises samples per split for very small datasets)
    stratify_by  column whose categories are balanced across splits
                 (e.g. "condition"); None skips stratification
    seed         RNG seed — fix for reproducible / shareable splits
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

    imgs = table.drop_duplicates(["replicate", "image_name"])
    groups = [imgs] if stratify_by is None else [g for _, g in imgs.groupby(stratify_by)]
    chosen = {"train": [], "val": [], "test": []}
    for g in groups:
        keys = g[["replicate", "image_name"]].to_numpy()
        keys = keys[rng.permutation(len(keys))]
        n = len(keys)
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

    Parameters
    ----------
    data_path           path to multinucleation.zarr
    cell_table_csv      path to per-cell CSV (built by scripts/build_cell_table.py)

    -- what to load --
    patch_type          "cPatches" (masked, isolated) or "bbPatches" (raw context)
    channels            channel indices — (0,)=membrane, (1,)=nuclei, (0,1)=both
    masks               extra mask arrays returned per sample, e.g. ("cCellmask",)

    -- dataloader --
    batch_size          cells per batch
    num_workers         parallel I/O workers; 0 = main process (useful to debug)
    pin_memory          pin CPU tensors for faster GPU transfer (leave True on GPU)
    persistent_workers  keep workers alive between epochs (recommended: True)
    prefetch_factor     batches prefetched per worker; None = PyTorch default

    -- splitting --
    split_ratios        (train, val, test) fractions, must sum to 1
    split_by            "image" (no FOV leakage, recommended) or "cell"
    stratify_by         table column to balance across splits (e.g. "condition")
    seed                RNG seed for reproducible splits

    -- normalisation --
    augment             apply geometric + photometric augmentation to train set
    norm_low            lower percentile for in-mask per-channel normalisation
    norm_high           upper percentile (patches on disk are raw uint16-range)
    norm_mask_array     zarr array used as the normalisation mask
    """

    def __init__(
        self,
        data_path,
        cell_table_csv,
        patch_type: str = "cPatches",
        channels: Sequence[int] | None = (0, 1),
        masks: Sequence[str] = (),
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
        norm_low: float = 1.0,
        norm_high: float = 99.0,
        norm_mask_array: str = "cCellmask",
    ):
        super().__init__()
        self.save_hyperparameters()     # Lightning: logs all constructor args
        self.data_path = str(data_path)
        self.cell_table_csv = str(cell_table_csv)
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
        self.augment = augment
        self.norm_low = norm_low
        self.norm_high = norm_high
        self.norm_mask_array = norm_mask_array

        # overridable after construction for custom pipelines
        self.train_transform: Callable | None = None
        self.val_transform: Callable | None = None

        self.table: pd.DataFrame | None = None
        self.splits: dict | None = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def _make_dataset(self, split_idx, transform) -> CellPatchDataset:
        return CellPatchDataset(
            self.data_path, split_idx, self.table,
            patch_type=self.patch_type, channels=self.channels,
            masks=self.masks, transform=transform,
            norm_mask_array=self.norm_mask_array,
        )

    def setup(self, stage: str | None = None):
        if self.table is None:
            self.table = load_cell_table(self.cell_table_csv)
        if self.splits is None:
            self.splits = build_splits(
                self.table, ratios=self.split_ratios, split_by=self.split_by,
                stratify_by=self.stratify_by, seed=self.seed,
            )
        train_tf = self.train_transform or build_transforms(self.augment, self.norm_low, self.norm_high)
        val_tf = self.val_transform or build_transforms(False, self.norm_low, self.norm_high)

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
        """Write train/val/test cell_idx lists to JSON (share to lock in the partition)."""
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

def compute_normalization_stats(data_path, table, cell_idx, patch_type="cPatches",
                                channels=(0, 1), low=1.0, high=99.0,
                                max_cells=2000, seed=0) -> dict:
    """Estimate global per-channel percentile / mean / std over a cell sample.

    Not required for the default per-cell normalisation; useful for bias
    analysis or designing an alternative global-normalisation scheme.
    Returns {"low":[…], "high":[…], "mean":[…], "std":[…]}.
    """
    root = zarr.open_group(str(data_path), mode="r")
    rng = np.random.default_rng(seed)
    idx = np.asarray(list(cell_idx), dtype=np.int64)
    if len(idx) > max_cells:
        idx = rng.choice(idx, size=max_cells, replace=False)

    ch = list(channels)
    acc = [[] for _ in ch]
    for ci in idx:
        row = table.loc[int(ci)]
        rep, cond, img, loc = (str(row["replicate"]), str(row["condition"]),
                               str(row["image_name"]), int(row["local_cell_index"]))
        arr = root[f"patches/{rep}/{cond}/{img}"][patch_type][loc]
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
