#%%
from darth_vaeder.datamodules.dataset_EG import BorderCellDataset, BCDataModule, percentile_norm, no_transform
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

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
#%%
transform_module = BCDataModule(dataset, 
                      spatial_transforms=v2.Compose([v2.RandomRotation(180), 
                                                    v2.RandomHorizontalFlip(p=1),
                                                    v2.RandomVerticalFlip(p=1), v2.RandomAffine(15), 
                                                    v2.RandomErasing(p=1, scale=(0.02,0.33))]), intensity_transforms= no_transform, batch_size=16, num_workers = 8)
#%%
# test transforms
transform_module.setup("fit")
train_dataloader = transform_module.train_dataloader()
batch = next(iter(train_dataloader))
val_dataloader = transform_module.val_dataloader()
batch = next(iter(val_dataloader))
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
import torch
import torch.nn.functional as F
import lightning as L
#%%
shape = (4, 2, 256, 256)

tensor1 = torch.rand(shape)
tensor2 = torch.rand(shape)

F.mse_loss(tensor1, tensor2, reduction='sum')/tensor1.shape[0]



# %%
