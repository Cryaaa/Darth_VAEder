import pandas as pd
import numpy as np
import tifffile


csv_path="/mnt/efs/dl_jrc/student_data/S-DA/image_metadata_BR00149208.csv"
df=pd.read_csv(csv_path)

# incrementing ID per field, 2000 ish in total 
df["Group_id"] =  df.groupby(['Row', 'Column', 'Field']).ngroup()

# Subset for a channel

channel_1_mask = df["Channel Index"] == 1
channel_1 = df[channel_1_mask]


all_square_means = []
all_means = []

for ID in channel_1["Group_id"]:

    mask = channel_1["Group_id"] == ID
    path = channel_1[mask]["Path"].iloc[0]

    img = tifffile.imread(path)
    img = img.astype(np.float64)

    img_sqr_mean = np.mean(img**2)
    all_square_means.append(img_sqr_mean)

    img_mean = np.mean(img)
    all_means.append(img_mean)
    break
    

all_square_means = np.array(all_square_means)
all_means = np.array(all_means)


overall_mean = np.mean(all_means)
variance = np.mean(all_square_means) - overall_mean**2

std = np.sqrt(variance)


class Mean_std_plate(): 
    def __init__(self, df: pd.DataFrame, channel_IDs:list = [1,2,3,4], ):
        if "Group_id" in df.columns:
            self.df = df
        else:
            self.df = self.add_group_id(df)

        
        self.channel_IDs = channel_IDs


    def compute_per_ch(channel_IDs):


    def compute():

        channel_1_mask = df["Channel Index"] == 1
        channel_1 = df[channel_1_mask]


        all_square_means = []
        all_means = []

        for ID in channel_1["Group_id"]:

            mask = channel_1["Group_id"] == ID
            path = channel_1[mask]["Path"].iloc[0]

            img = tifffile.imread(path)
            img = img.astype(np.float64)

            img_sqr_mean = np.mean(img**2)
            all_square_means.append(img_sqr_mean)

            img_mean = np.mean(img)
            all_means.append(img_mean)
            break
            

        all_square_means = np.array(all_square_means)
        all_means = np.array(all_means)


        overall_mean = np.mean(all_means)
        variance = np.mean(all_square_means) - overall_mean**2

        std = np.sqrt(variance)

        return
    

    def add_group_id(self, df): 
        # incrementing ID per field
        df["Group_id"] =  df.groupby(['Row', 'Column', 'Field']).ngroup()
        return df


