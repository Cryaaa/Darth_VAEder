import pandas as pd
from torch.utils.data import Dataset
import zarr
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        zarr_path,
        input_array_name,
        input_mask_name
    ):
        self.metadata = pd.read_csv(annotations_file)
        self.img_dir = zarr_path
        zarr_group = zarr.open(zarr_path)
        self.input_array_name = input_array_name
        self.input_mask_name = input_mask_name
        # TODO
        # get all relevant images and annotations and load them
        # into memory
        self.inputs = []
        self.masks = []
        for i, row in self.metadata.iterrows():
            name = str(row["image_id"])
            zarr_sample = zarr_group[name]
            self.inputs.append(np.array(zarr_sample[input_array_name]))
            self.masks.append(np.array(zarr_sample[input_mask_name]))

    def __len__(self):
        # Figure out the total number of samples in the dataset and return it
        return len(self.inputs)

    def __getitem__(self, idx):
        # TODO
        # Given an index, return the corresponding sample from the dataset.
        # This will typically involve:
        # 1. Loading the image from disk using the file path
        # 2. Applying any necessary transformations to the image
        # (e.g., resizing, normalization) if not already done in memory
        # 3. loading the metadata for the image
        # 4. adding a target image
        # 5. (optionally) return a mask
        output = {
            "source": self.inputs[idx],
            "target": self.inputs[idx],
            "masks": self.masks[idx],
            "metadata": self.metadata.iloc[idx],
        }
        return output
