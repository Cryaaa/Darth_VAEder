import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import tifffile as tiff


csv_path="/mnt/efs/dl_jrc/student_data/S-DA/image_metadata_BR00149208.csv"
df=pd.read_csv(csv_path)
df["Group_id"] =  df.groupby(['Row', 'Column', 'Field']).ngroup()


class CustomImageDataset(Dataset):
    def __init__(
        self,
        metadata_file,
        annotations_file,
    ):
        
        self.df = df
        self.img_labels = pd.read_csv(annotations_file)
        self.metadata_file=metadata_file
   #     self.img_dir = img_dir
   #     self.transform = transform

        # get all relevant images and annotations and load them
        # into memory

    def __len__(self):
        # Figure out the total number of samples in the dataset and return it
        # TODO
        return
    
    
     # Center crop
     # load image

        center_cropped_img = self.centercrop(...)
        for channel in range(center_cropped_img.shape[]
            self.compute_normalization(img)   
    
    # compute normalization 
    

    def preprocess_normalization(self,img) ->dict:
        #load image and normalization by robust Z


        return{
            "mean":...,
            "median":...,
            "std":...,
            "iqr":...
        }
    


       

        # add the statistic columns ( mean median, etc)


        #save the new metadata file with stats

        return 

    
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

