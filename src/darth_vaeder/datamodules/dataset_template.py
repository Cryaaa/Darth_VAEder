import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import tifffile as tiff


csv_path="/mnt/efs/dl_jrc/student_data/S-DA/image_metadata_BR00149208.csv"
df=pd.read_csv(csv_path)

# incrementing ID per field, 2000 ish in total 
df["Group_id"] =  df.groupby(['Row', 'Column', 'Field']).ngroup()

# incrementing ID per well, 60 in total 
df["Well_id"] =  df.groupby(['Row', 'Column']).ngroup()

# get unique well IDs 
unique_well_IDs = df["Well_id"].unique()

# percentage of fields to sample from each well 
n_sample = 5 # as % 
n = int((n_sample * len(unique_well_IDs)) / 100)



fields_to_draw = []
seed = 42
rng = np.random.default_rng(seed)

for well in unique_well_IDs:
    mask = df["Well_id"] == well

    well_df = df[mask]

    unique_field_ids = well_df["Field"].unique()
    fields = rng.choice(unique_field_ids, size = n, replace = False)
    field_mask = well_df["Field"].isin(fields)
    
    unique_fields = well_df[field_mask]["Group_id"].unique()
    fields_to_draw.append(unique_fields)


fields_to_draw = np.concatenate(fields_to_draw)



all_paths = []
#TODO: we are currently ding this for field 0. Do this for all of our fields in fields_to_draw
mask = (df["Group_id"] == fields_to_draw[0]) & (df["Channel Index"] == 1)

all_paths.append(df[mask]["Path"].iloc[0])










def get_norm_stats(df):









class CustomImageDataset(Dataset):
    def __init__(
        self,
        metadata_file,
    ):
        
        self.metadata_file=metadata_file


    def __len__(self):
        # Figure out the total number of samples in the dataset and return it
        # TODO
        return
    
    
     # Center crop
     # load image

    
    # compute normalization 
    

    # def preprocess_normalization(self,img) ->dict:
    #     #load image and normalization by robust Z


    #     return{
    #         "mean":...,
    #         "median":...,
    #         "std":...,
    #         "iqr":...
    #     }

    #     # add the statistic columns ( mean median, etc)


    #     #save the new metadata file with stats

    #     return 

    
    def centercrop(self, img, sidelength = 256):
        
        center = np.array(img.shape[1:]) // 2
        y, x = center - sidelength // 2

        return img[:, y: y + sidelength, x : x + sidelength]





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
        group = df[df['Group_id'] == idx]

        group = group.sort_values("Channel Index")
        
        # normalize
        #per well normalization


        stacked_img = torch.from_numpy(
            np.stack([tiff.imread(path) for path in group["Path"]])
        )
        cropped_image=self.centercrop(stacked_img)
        


        



        output = {
            "source": ...,
            "target": ...,
            "metadata": ...,
        }
        return output



getdataset = CustomImageDataset(df)




# 

        # center_cropped_img = self.centercrop(...)
        # for channel in range(center_cropped_img.shape[]
        #     self.compute_normalization(img)   