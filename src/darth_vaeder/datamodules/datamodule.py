"""LightningDataModule wrapping CellDataset for train / val / test splits.

Splits are assigned by image (not by cell) to avoid data leakage —
cells from the same image are correlated.
"""

import lightning as L


class MultinucDataModule(L.LightningDataModule):
    """Manages CellDataset splits and DataLoaders.

    Parameters
    ----------
    zarr_path:
        Path to multinucleation.zarr store.
    splits:
        Dict with keys 'train', 'val', 'test', each a list of ncells_idx.
    stats:
        Per-channel mean/std dict (computed by scripts/compute_stats.py).
    batch_size:
        Number of cells per batch.
    num_workers:
        DataLoader worker processes.
    use_context:
        If True, use bbPatches (context window) instead of cPatches.
    """

    def __init__(self, zarr_path, splits, stats, batch_size=128, num_workers=8, use_context=False):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
