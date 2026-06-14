# %% [markdown]
# ## Datamodule & Dataset — interactive walkthrough
# Run cell-by-cell in VS Code (Python Interactive) or Jupyter.
# Adjust ZARR_PATH / TABLE_PATH to your server paths.

# %%
ZARR_PATH  = "/mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr"
TABLE_PATH = "/mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/outputs/cell_table.csv"

# ──────────────────────────────────────────────────────────────
# 1. DataModule  — splits by FOV, stratified by condition
# ──────────────────────────────────────────────────────────────
# %%
from darth_vaeder.datamodules import MultinucDataModule

dm = MultinucDataModule(
    data_path=ZARR_PATH,
    cell_table_csv=TABLE_PATH,
    patch_type="cPatches",          # isolated, masked cell
    channels=(0, 1),                # 0=membrane  1=nuclei
    masks=("cCellmask", "cNucmask"),  # returned per sample for loss / viz
    batch_size=32,
    num_workers=4,
    persistent_workers=True,        # keep workers alive across epochs
)
dm.setup("fit")

print("Split sizes:", {k: len(v) for k, v in dm.splits.items()})
print("Table columns:", list(dm.table.columns))

# ──────────────────────────────────────────────────────────────
# 2. One training batch
# ──────────────────────────────────────────────────────────────
# %%
train_loader = dm.train_dataloader()

batch = next(iter(train_loader))
print("image  :", batch["image"].shape,   batch["image"].dtype)    # (B, 2, 256, 256)
print("mask   :", batch["cCellmask"].shape)                        # (B, 1, 256, 256)
print("range  :", batch["image"].min().item(), "→", batch["image"].max().item())
print("conds  :", sorted(set(batch["metadata"]["condition"])))

# ──────────────────────────────────────────────────────────────
# 3. Dataset directly — address specific cells by cell_idx
# ──────────────────────────────────────────────────────────────
# %%
from darth_vaeder.datamodules import CellPatchDataset, load_cell_table

table = load_cell_table(TABLE_PATH)

dataset = CellPatchDataset(
    data_path=ZARR_PATH,
    cell_idx=[0, 1, 2, 3, 4],          # any list of cell_idx integers
    table=table,
    patch_type="cPatches",
    channels=(0, 1),
    masks=("cCellmask",),
    transform=None,                     # raw, unnormalised for inspection
    norm_mask_array="cCellmask",
)

sample = dataset[0]
print("keys  :", list(sample.keys()))
print("image :", sample["image"].shape, "  range:", sample["image"].min().item(), "→", sample["image"].max().item())
print("meta  :", sample["metadata"])

# ──────────────────────────────────────────────────────────────
# 4. Visualise a batch  (membrane / nuclei / mask overlay)
# ──────────────────────────────────────────────────────────────
# %%
import matplotlib.pyplot as plt
import numpy as np

def show_cells(batch, n=8):
    imgs  = batch["image"][:n].cpu().numpy()       # (N, C, H, W) in [0,1]
    masks = batch["cCellmask"][:n, 0].cpu().numpy() > 0  # (N, H, W) bool

    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 7))
    for i in range(n):
        mem, nuc = imgs[i, 0], imgs[i, 1]
        ovl = np.stack([mem, mem, mem], axis=-1)
        ovl[masks[i], 0] = np.clip(ovl[masks[i], 0] + 0.4, 0, 1)  # red in-mask

        axes[0, i].imshow(mem, cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(nuc, cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(ovl)

        cond = batch["metadata"]["condition"][i]
        axes[0, i].set_title(cond, fontsize=7)

    for ax, lbl in zip(axes[:, 0], ["membrane", "nuclei", "mask ovl"]):
        ax.set_ylabel(lbl, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    plt.show()

show_cells(batch, n=min(8, batch["image"].shape[0]))

# %%
