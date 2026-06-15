#%%
from darth_vaeder.datamodules.per_plate_normalization import Mean_std_plate
import pandas as pd
import json

csv_path="/mnt/efs/dl_jrc/student_data/S-DA/image_metadata_BR00149208.csv"
df=pd.read_csv(csv_path)

stats = Mean_std_plate(df).compute_per_ch()
#%%
stats
#%%
ouput_file = "/mnt/efs/dl_jrc/student_data/S-DA/data/BR00149208__2026-01-22T17_14_56-Measurement 1/per_plate_mean_stddev_per_channel.json"
with open(ouput_file,'w') as f:
    json.dump(stats, f)
# %%
