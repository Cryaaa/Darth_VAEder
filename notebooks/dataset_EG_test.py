#%%
from darth_vaeder.datamodules.dataset_EG import BorderCellDataset

#%%
dataset = BorderCellDataset(
    "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv", 
    "/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr", "max_projection", 
    "3D_mask_corrected"
    )


# %%
dataset[0]["metadata"]

# %%
