#%%
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.io import imread

csv_path="/mnt/efs/dl_jrc/student_data/S-DA/image_metadata_BR00149208.csv"
df=pd.read_csv(csv_path)

i=0
image_path_i=df.iloc[i]["Path"]
image_i=imread(image_path_i)

plt.imshow(image_i)
# %%
