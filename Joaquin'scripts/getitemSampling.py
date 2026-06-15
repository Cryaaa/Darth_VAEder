#%%
"""Sample one training batch exactly as train.py does and print diagnostics.

Run:
    python "Joaquin'scripts/debug_batch.py" \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv
"""

import argparse
import torch
import matplotlib.pyplot as plt
from darth_vaeder.datamodules import MultinucDataModule

# p = argparse.ArgumentParser()
# p.add_argument("--zarr",    required=True)
# p.add_argument("--table",   required=True)
# p.add_argument("--batch",   type=int, default=8)
# p.add_argument("--workers", type=int, default=0)
# args = p.parse_args()

# ── identical to train.py ────────────────────────────────────────────────
dm = MultinucDataModule(
    data_path="/mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr",
    cell_table_csv="outputs/cell_table.csv",
    batch_size=8,
    num_workers=0,
)
    
dm.setup("fit")
train_dataloader = dm.train_dataloader()
    # batch = next(iter(dm.train_dataloader()))
# %%
loader = dm.train_dataloader()  # or dm.val_dataloader()/dm.test_dataloader()
for batch_idx, batch in enumerate(train_dataloader):
    x_img = batch["cPatch"]      # (B, C, H, W)
    mask  = batch["pCellmask"]   # (B, 1, H, W)
    meta  = batch["metadata"]
    plt.imshow(x_img[0, 0].numpy())
    print(x_img.min(), x_img.max())
    plt.show()
    # plt.imshow(mask[0, 0].numpy(), alpha=0.5)
    # plt.show()
    # process batch...

    if batch_idx >10:
        break

# %%
