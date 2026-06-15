#%%
from darth_vaeder.datamodules.HTSdatamodule import CustomImageDataset, HTSDataModule
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, Compose

csv_path="/mnt/efs/dl_jrc/student_data/S-DA/image_metadata_BR00149208.csv"
json_path = "/mnt/efs/dl_jrc/student_data/S-DA/data/BR00149208__2026-01-22T17_14_56-Measurement 1/per_plate_mean_stddev_per_channel.json"

dataset = CustomImageDataset(
    csv_path,
    json_path,
)
# %%
import json

with open(json_path, 'r') as f:
    stats = json.load(f)

stats
# %%
import matplotlib.pyplot as plt
import numpy as np
dataset.spatial_transforms = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
])

plt.imshow(np.transpose(dataset[0]["source"].numpy()[:3],(1,2,0)))
# %%
