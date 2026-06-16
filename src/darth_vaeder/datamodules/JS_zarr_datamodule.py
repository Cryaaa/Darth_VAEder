"""Lightning data module + dataset for the multinucleation.zarr single-cell store.

Each training sample is one cell.  The dataset loads cPatch + pCellmask;
bbPatch is optional (include_bb=True) for later model variants.

    cPatch      (C, H, W)  float32   isolated masked cell, raw
    pCellmask   (1, H, W)  int64     dilated crop mask (falls back to cCellmask)
    bbPatch     (C, H, W)  float32   context window, raw (only if include_bb=True)

Normalisation and augmentation live in transforms.py.  Nothing is pre-normalised
on disk.  Routing uses cell_table.csv (built by scripts/build_cell_table.py).

Quick start
-----------
    dm = MultinucDataModule(
        data_path="/.../multinucleation.zarr",
        cell_table_csv="outputs/cell_table.csv",
        include_bb=False,          # first run: cPatch + cCellmask only
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    # batch["cPatch"]    (B, C, 256, 256)
    # batch["pCellmask"] (B, 1, 256, 256)
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

from .JS_transforms import build_train_transforms, build_val_transforms


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


def load_cell_table(cell_table_csv) -> pd.DataFrame:
    """Read cell_table.csv and index it by cell_idx (column kept too)."""
    return pd.read_csv(cell_table_csv).set_index("cell_idx", drop=False)


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class CellPatchDataset(Dataset):
    """One zarr chunk (one cell) per item.

    Parameters
    ----------
    data_path   path to multinucleation.zarr
    cell_idx    cell_idx values this split owns
    table       full cell table from load_cell_table()
    channels    which image channels to load — (0,)=membrane, (1,)=nuclei, (0,1)=both
    include_bb  also load bbPatch (context window); False for the first run
    transform   sample-dict transform (normalise + augment, see transforms.py)
    """

    def __init__(
        self,
        data_path,
        cell_idx: Sequence[int],
        table: pd.DataFrame,
        channels: Sequence[int] | None = (0, 1),
        include_bb: bool = False,
        transform: Callable | None = None,
    ):
        self.data_path  = str(data_path)
        self.cell_idx   = np.asarray(cell_idx, dtype=np.int64)
        self.table      = table
        self.channels   = list(channels) if channels is not None else None
        self.include_bb = include_bb
        self.transform  = transform
        self._root      = None   # opened lazily per worker (zarr not fork-safe)

    def _ensure_open(self):
        if self._root is None:
            self._root = zarr.open_group(self.data_path, mode="r")

    def __len__(self) -> int:
        return len(self.cell_idx)

    def _load_raw(self, i: int) -> dict:
        """Load one cell from zarr and return the raw sample dict (no transform, no metadata)."""
        self._ensure_open()
        ci  = int(self.cell_idx[i])
        row = self.table.loc[ci]
        rep, cond = str(row["replicate"]), str(row["condition"])
        img, loc  = str(row["image_name"]), int(row["local_cell_index"])
        pg = self._root[f"patches/{rep}/{cond}/{img}"]

        def _img(key):
            arr = np.ascontiguousarray(pg[key][loc])          # (H, W, C)
            t   = torch.from_numpy(arr).permute(2, 0, 1).float()
            return t[self.channels] if self.channels is not None else t

        def _mask(key):
            arr = np.ascontiguousarray(pg[key][loc])          # (H, W)
            return torch.from_numpy(arr).long().unsqueeze(0)  # (1, H, W)

        sample = {
            "cPatch":     _img("cnPatches" if "cnPatches" in pg else "cPatches"),
            "pCellmask":  _mask("pCellmask" if "pCellmask" in pg else "cCellmask"),
            "pNucmask":   _mask("pNucmask"  if "pNucmask"  in pg else "cNucmask"),
            "norm_stats": {
                "mem_lo": float(row["norm_mem_lo"]),
                "mem_hi": float(row["norm_mem_hi"]),
                "nuc_lo": float(row["norm_nuc_lo"]),
                "nuc_hi": float(row["norm_nuc_hi"]),
            },
            "index": ci,
        }
        if self.include_bb:
            sample["bbPatch"]    = _img("bbPatches")
            sample["bbCellmask"] = _mask("bbCellmask")
        return sample

    def __getitem__(self, i: int) -> dict:
        sample = self._load_raw(i)

        if self.transform is not None:
            sample = self.transform(sample)

        ci  = int(self.cell_idx[i])
        row = self.table.loc[ci]
        sample["metadata"] = {k: _py(v) for k, v in row.to_dict().items()}
        return sample


def vae_collate(batch: list[dict]) -> dict:
    """Stack tensors; keep metadata without converting strings to tensors."""
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
    """Partition cell_idx into train / val / test.

    split_by="image" keeps all cells from one FOV together (recommended —
    prevents evaluating on cells from FOVs seen during training).
    stratify_by balances the chosen column (e.g. "condition") across splits.
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
    """Lightning wrapper for CellPatchDataset.

    Parameters
    ----------
    data_path           path to multinucleation.zarr
    cell_table_csv      path to cell_table.csv (scripts/build_cell_table.py)

    -- inputs --
    channels            channel indices (0=membrane, 1=nuclei, None=all)
    include_bb          load bbPatch (context window); False for first run

    -- dataloader --
    batch_size, num_workers, pin_memory, persistent_workers, prefetch_factor
                        standard DataLoader knobs

    -- splitting --
    split_ratios        (train, val, test) — must sum to 1
    split_by            "image" (recommended) or "cell"
    stratify_by         column to balance across splits (e.g. "condition")
    seed                RNG seed for reproducible splits

    -- augmentation / normalisation --
    augment             True → normalise + rotate360 + hflip + vflip (train set)
                        False → normalise only (val/test behaviour applied to train too)
    norm_mask           sample-dict key used for percentile-normalisation stats.
                        "pCellmask" (default) = dilated crop mask, covers the full
                        cell content in cPatches.  "cCellmask" = tight boundary.
    norm_low/high       percentile bounds for NormalizeMasked (cPatch, in-mask pixels)
    """

    def __init__(
        self,
        data_path,
        cell_table_csv,
        channels: Sequence[int] | None = (0, 1),
        include_bb: bool = False,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
        split_ratios: tuple = (0.7, 0.15, 0.15),
        split_by: str = "image",
        stratify_by: str | None = "condition",
        seed: int = 42,
        augment: bool = True,
        norm_mask:      str   = "pCellmask",
        cell_norm_low:  float = 1.0,
        cell_norm_high: float = 99.0,
        img_size:       int   = 256,   # [256]: no img_size param; set to 96 for downsampled mode
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path          = str(data_path)
        self.cell_table_csv     = str(cell_table_csv)
        self.channels           = channels
        self.include_bb         = include_bb
        self.batch_size         = batch_size
        self.num_workers        = num_workers
        self.pin_memory         = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor    = prefetch_factor
        self.split_ratios       = split_ratios
        self.split_by           = split_by
        self.stratify_by        = stratify_by
        self.seed               = seed
        self.augment            = augment
        self.norm_mask          = norm_mask
        self.cell_norm_low      = cell_norm_low
        self.cell_norm_high     = cell_norm_high
        self.img_size           = img_size

        # overridable after construction for custom pipelines
        self.train_transform: Callable | None = None
        self.val_transform:   Callable | None = None

        self.table: pd.DataFrame | None = None
        self.splits: dict | None        = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    @property
    def _image_keys(self):
        return ("cPatch", "bbPatch") if self.include_bb else ("cPatch",)

    def _make_dataset(self, split_idx, transform) -> CellPatchDataset:
        return CellPatchDataset(
            self.data_path, split_idx, self.table,
            channels=self.channels, include_bb=self.include_bb,
            transform=transform,
        )

    def setup(self, stage: str | None = None):
        if self.table is None:
            self.table = load_cell_table(self.cell_table_csv)
        if self.splits is None:
            self.splits = build_splits(
                self.table, ratios=self.split_ratios, split_by=self.split_by,
                stratify_by=self.stratify_by, seed=self.seed,
            )
        train_tf = self.train_transform or (
            # [256]: build_train_transforms(self._image_keys, mask_keys=("pCellmask",), norm_mask=self.norm_mask)
            build_train_transforms(self._image_keys, mask_keys=("pCellmask",),
                                   norm_mask=self.norm_mask, img_size=self.img_size)
            if self.augment else build_val_transforms(norm_mask=self.norm_mask, img_size=self.img_size)
        )
        # [256]: val_tf = self.val_transform or build_val_transforms(norm_mask=self.norm_mask)
        val_tf = self.val_transform or build_val_transforms(norm_mask=self.norm_mask, img_size=self.img_size)
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
        """Persist train/val/test cell_idx lists to JSON."""
        if self.splits is None:
            raise RuntimeError("Call setup() first.")
        with open(path, "w") as f:
            json.dump({k: np.asarray(v).tolist() for k, v in self.splits.items()}, f)

    def load_splits(self, path):
        """Load a saved JSON partition before setup() to override auto-splitting."""
        with open(path) as f:
            self.splits = {k: np.asarray(v, dtype=np.int64) for k, v in json.load(f).items()}
