#%%
from darth_vaeder.datamodules.dataset_EG import BorderCellDataset, BCDataModule, percentile_norm
import matplotlib.pyplot as plt
import numpy as np
import torch

#%%
dataset = BorderCellDataset(
    "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv", 
    "/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr", "max_projection", 
    "3D_mask_corrected"
    )


# %%
# Debugging efforts
print(f"len of dataset{len(dataset)}")
idx = 100
source = dataset[idx]["source"]
target = dataset[idx]["target"]
mask = dataset[idx]["masks"]

plt.imshow(mask[0])


# %%
# Train the data module
train_module = BCDataModule(dataset, percentile_norm, None, batch_size = 4)
# %%
train_module.setup("fit")
train_dataloader = train_module.train_dataloader()
batch = next(iter(train_dataloader))
val_dataloader = train_module.val_dataloader()
batch = next(iter(val_dataloader))
#%%

# %%
plt.imshow(batch["source"][0,1])
print(batch['source'][0,0].mean())
# %%
import torch

test = torch.Tensor((2,2)).to(bool).to(float)
test
# %%
