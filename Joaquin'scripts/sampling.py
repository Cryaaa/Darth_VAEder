#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "/home/S-JS/Darth_VAEder/src")

from darth_vaeder.datamodules.JS_zarr_datamodule import CellPatchDataset, load_cell_table, build_splits
from darth_vaeder.datamodules.JS_transforms import build_val_transforms, build_train_transforms

ZARR  = "/mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr"
TABLE = "/mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/outputs/cell_table.csv"
N     = 8  # cells to show

# ── load table and pick N random cells ───────────────────────────────────────
table  = load_cell_table(TABLE)
splits = build_splits(table)
idx    = np.random.choice(splits["train"], size=N, replace=False)

# ── three datasets: raw, normalized, augmented ────────────────────────────────
ds_raw  = CellPatchDataset(ZARR, idx, table, transform=None)
ds_norm = CellPatchDataset(ZARR, idx, table, transform=build_val_transforms())
ds_aug  = CellPatchDataset(ZARR, idx, table, transform=build_train_transforms(("cPatch",)))

def norm01(t):
    mn, mx = t.min(), t.max()
    return ((t - mn) / (mx - mn + 1e-6)).numpy()

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(N, 6, figsize=(14, N * 2.2))

titles = ["raw ch0", "raw ch1", "norm ch0", "norm ch1", "aug ch0", "aug ch1"]
for j, t in enumerate(titles):
    axes[0, j].set_title(t, fontsize=9)

for i in range(N):
    raw  = ds_raw[i]["cPatch"]
    norm = ds_norm[i]["cPatch"]
    aug  = ds_aug[i]["cPatch"]

    for col, patch in enumerate([raw, norm, aug]):
        for ch in range(2):
            ax = axes[i, col * 2 + ch]
            ax.imshow(norm01(patch[ch]), cmap="gray", interpolation="nearest")
            ax.axis("off")

plt.tight_layout()
plt.show()
# %%
