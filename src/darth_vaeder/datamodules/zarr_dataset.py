"""Dataset that reads single-cell patches from the multinucleation.zarr store.

One sample = one valid cell (indexed by global ncells_idx).
Reads cPatches or bbPatches + corresponding masks from the zarr hierarchy.
Zarr handle is opened lazily per worker to avoid fork-safety issues with DataLoader.
"""

from torch.utils.data import Dataset


class CellDataset(Dataset):
    """Zarr-backed dataset for cardiomyocyte cell patches.

    Parameters
    ----------
    zarr_path:
        Path to multinucleation.zarr store.
    indices:
        Array of global ncells_idx values this split owns.
    stats:
        Dict with keys 'mean' and 'std', shape (2,), for per-channel normalisation.
    transform:
        Optional callable applied to (image, mask) after loading.
    use_context:
        If True, load bbPatches (full context window) instead of cPatches (isolated cell).
    """

    def __init__(self, zarr_path, indices, stats, transform=None, use_context=False):
        pass

    def __len__(self):
        pass

    def __getitem__(self, i):
        # Returns dict with keys: image (2,256,256), mask (256,256),
        # ncells_idx, condition, replicate
        pass
