import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import tifffile as tiff
import json

# csv_path="/mnt/efs/dl_jrc/student_data/S-DA/image_metadata_BR00149208.csv"
# df=pd.read_csv(csv_path)

# # incrementing ID per field, 2000 ish in total 
# df["Group_id"] =  df.groupby(['Row', 'Column', 'Field']).ngroup()

# # incrementing ID per well, 60 in total 
# df["Well_id"] =  df.groupby(['Row', 'Column']).ngroup()

# # get unique well IDs 
# unique_well_IDs = df["Well_id"].unique()

# # percentage of fields to sample from each well 
# n_sample = 5 # as % 
# n = int((n_sample * len(unique_well_IDs)) / 100)



# fields_to_draw = []
# seed = 42
# rng = np.random.default_rng(seed)

# for well in unique_well_IDs:
#     mask = df["Well_id"] == well

#     well_df = df[mask]

#     unique_field_ids = well_df["Field"].unique()
#     fields = rng.choice(unique_field_ids, size = n, replace = False)
#     field_mask = well_df["Field"].isin(fields)
    
#     unique_fields = well_df[field_mask]["Group_id"].unique()
#     fields_to_draw.append(unique_fields)


# fields_to_draw = np.concatenate(fields_to_draw)



# all_paths = [] 
# # for channel_idx in range(1,5):
# for i in fields_to_draw,
# #TODO: we are currently ding this for field 0. Do this for all of our fields in fields_to_draw
# mask = (df["Group_id"] == fields_to_draw[i]) & (df["Channel Index"] == 1)

# all_paths.append(df[mask]["Path"].iloc[0])










# def get_norm_stats(df):
#     pass





def no_transform(input):
    return input

def centercrop(img, sidelength = 256):
    center = np.array(img.shape)[[-2,-1]] // 2
    y, x = center - sidelength // 2

    return img[..., y: y + sidelength, x : x + sidelength]

class CustomImageDataset(Dataset):
    def __init__(
        self,
        metadata_file: str,
        pp_norm_json: str,
        spatial_transforms=no_transform,
        intensity_transforms=no_transform,
    ):
        dataframe = pd.read_csv(metadata_file)
        # incrementing ID per field, 2000 ish in total 
        dataframe["Group_id"] =  dataframe.groupby(['Row', 'Column', 'Field']).ngroup()
        # incrementing ID per well, 60 in total 
        dataframe["Well_id"] =  dataframe.groupby(['Row', 'Column']).ngroup()
        self.metadata_table = dataframe

        self.spatial_transforms = spatial_transforms
        self.intensity_transforms = intensity_transforms

        
        with open(pp_norm_json) as f:
            normalization_stats = json.load(f)
        channel_indices = dataframe["Channel Index"].unique()
        print(channel_indices)
        image_array = []
        group_ids =[]
        for group_id in dataframe["Group_id"].unique():
            subset_data = dataframe[dataframe["Group_id"] == group_id]
            multichannel = []
            for ch_idx in channel_indices:
                row = subset_data[subset_data["Channel Index"]==ch_idx]
                # print(row)
                file_path = row["Path"].iloc[0]
                img = tiff.imread(file_path)
                crop = centercrop(img)
                #  TODO maybe change to median
                mean = normalization_stats[str(ch_idx)]["mean"]
                stdd = normalization_stats[str(ch_idx)]["std"]
                norm = (crop - mean) / stdd
                multichannel.append(norm)
            image_array.append(multichannel)
            group_ids.append(group_id)
        self.group_ids = group_ids
        self.image_array = image_array
                


    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, idx):
        group = self.group_ids[idx]
        image = self.image_array[idx]
        out_metadata = self.metadata_table[self.metadata_table["Group_id"]==group]
        
        image_tensor = torch.Tensor(image)
        target_tensor = torch.Tensor(image)
        image_tensor = self.spatial_transforms(image_tensor)
        image_tensor = self.intensity_transforms(image_tensor)
        
        target_tensor = self.spatial_transforms(target_tensor)


        output = {
            "source": image_tensor,
            "target": target_tensor,
            "metadata": out_metadata,
        }
        return output


import lightning as L
from torch.utils.data import random_split, DataLoader




class HTSDataModule(L.LightningDataModule):
    def __init__(
            self, 
            dataset_class,batch_size, 
            spatial_transforms, 
            intensity_transforms
        ):
        super().__init__()
        self.dataset = dataset_class
        self.batch_size = batch_size
        self.spatial_transforms = spatial_transforms
        self.intensity_transforms = intensity_transforms

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.ds_train, self.ds_val = random_split(
                self.dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
            self.ds_train.spatial_transforms = self.spatial_transforms
            # self.ds_train.intensity_transforms = ...

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            # TODO add second plate
            self.ds_test = self.dataset

        if stage == "predict":
            self.ds_pred = self.dataset

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.ds_pred, batch_size=self.batch_size)