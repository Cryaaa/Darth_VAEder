import pandas as pd
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        # TODO
        # get all relevant images and annotations and load them
        # into memory

    def __len__(self):
        # Figure out the total number of samples in the dataset and return it
        # TODO
        return

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
            "source": ...,
            "target": ...,
            "metadata": ...,
        }
        return output
