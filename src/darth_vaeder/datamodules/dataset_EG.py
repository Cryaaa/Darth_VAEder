import pandas as pd
from torch.utils.data import Dataset
from skimage import exposure
import zarr
import numpy as np
import lightning as L
import torch
from torch.utils.data import random_split, DataLoader
import warnings

def percentile_norm(image):
    p_low, p_high = np.percentile(image, (1, 99))
    scaled_image = exposure.rescale_intensity(image, in_range=(p_low, p_high), out_range=(0.0, 1.0))
    return scaled_image

def no_transform(input):
    return input

class BorderCellDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        zarr_path,
        input_array_name,
        input_mask_name,
        spatial_transforms = no_transform,
        intensity_transforms = no_transform,
        normalization_function = percentile_norm
    ):
        self.metadata = pd.read_csv(annotations_file)
        self.img_dir = zarr_path
        zarr_group = zarr.open(zarr_path)
        self.input_array_name = input_array_name
        self.input_mask_name = input_mask_name
        self.spatial_transforms = spatial_transforms
        self.intensity_transforms = intensity_transforms
        # TODO
        # get all relevant images and annotations and load them
        # into memory
        self.inputs = []
        self.masks = []
        for i, row in self.metadata.iterrows():
            name = str(row["image_id"])
            zarr_sample = zarr_group[name]
            image = np.array(zarr_sample[input_array_name])
            image = normalization_function(image)
            self.inputs.append(image)
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
        inputsT = torch.tensor(self.inputs[idx])
        masksT = torch.tensor(self.masks[idx])
        targetT = torch.tensor(self.inputs[idx])

        inputsT = self.spatial_transforms(inputsT)
        masksT = self.spatial_transforms(masksT)
        targetT = self.spatial_transforms(targetT)

        inputsT = self.intensity_transforms(inputsT)

        output = {
            "source": inputsT,
            "target": targetT,
            "masks": masksT,
            "metadata": self.metadata.iloc[idx],
        }
        return output


class BCDataModule(L.LightningDataModule):
    def __init__(self, dataset, spatial_transforms, intensity_transforms, batch_size):
        super().__init__()
        self.spatial_transforms = spatial_transforms
        self.intensity_transforms = intensity_transforms 
        self.batch_size = batch_size
        self.dataset = dataset
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            bc_full = self.dataset
            self.bc_train, self.bc_val = random_split(
                bc_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
            self.bc_train.spatial_transforms = self.spatial_transforms
            self.bc_train.intensity_transforms = self.intensity_transforms

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            warnings.Warn("no test ds implemented")
            self.bc_test = self.dataset

        if stage == "predict":
            self.bc_predict = self.dataset

    def train_dataloader(self):
        return DataLoader(self.bc_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.bc_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.bc_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.bc_predict, batch_size=self.batch_size)