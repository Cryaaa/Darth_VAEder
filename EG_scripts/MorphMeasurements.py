"""
Iterate through 3D binary masks stored in a Zarr store, collapse each mask
to a 2D max-intensity projection, compute 2D morphology metrics on the
projection, and write them out as extra columns on a copy of the metadata
CSV.

Expected Zarr layout:
    <zarr_root>/<image_id>/3D_mask_corrected   -> 3D binary array

Expected CSV layout:
    a column named "image_id" whose values match the zarr group names.
"""

import numpy as np
import pandas as pd
import zarr
from skimage.measure import label, regionprops
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG — edit these for your setup
# ---------------------------------------------------------------------------
ZARR_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/40xbordercell_dataset.zarr"
CSV_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x.csv"
OUTPUT_CSV_PATH = "/mnt/efs/dl_jrc/student_data/S-EG/project/data_information_PLC40x_with_morphology.csv"
MASK_KEY = "3D_mask_corrected"     # array name inside each image_id group

# Axis index of the z dimension in the mask array, e.g. shape (Z, Y, X) -> 0.
# Check the printed "mask shape" output the first time you run this and
# adjust if your array is ordered differently (e.g. (Y, X, Z) -> 2).
Z_AXIS = 0

# Physical size of one xy pixel (assumes square pixels). Leave as 1.0 to
# work in pixel units.
PIXEL_SIZE_XY = 1.0
# ---------------------------------------------------------------------------

def compute_morphology_2d(mask_2d: np.ndarray, pixel_size: float = 1.0) -> dict:
    """Compute 2D morphology metrics for a single-object binary mask."""
    metrics = {}
 
    pixel_count = int(mask_2d.sum())
    metrics["n_pixels"] = pixel_count
 
    empty_keys = [
        "area", "perimeter", "circularity", "solidity", "convex_area",
        "equivalent_diameter", "eccentricity", "major_axis_length",
        "minor_axis_length", "orientation", "extent"
    ]
    if pixel_count == 0:
        metrics.update({k: np.nan for k in empty_keys})
        return metrics
 
    props = regionprops(mask_2d.astype(np.uint8), spacing=(pixel_size, pixel_size))[0]
 
    area = props.area
    perimeter = props.perimeter
    metrics["area"] = area
    metrics["perimeter"] = perimeter
    metrics["circularity"] = 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else np.nan
 
    metrics["solidity"] = props.solidity
    metrics["convex_area"] = getattr(props, "area_convex", getattr(props, "convex_area", np.nan))
    metrics["equivalent_diameter"] = getattr(
        props, "equivalent_diameter_area", getattr(props, "equivalent_diameter", np.nan)
    )
    metrics["eccentricity"] = props.eccentricity
    metrics["major_axis_length"] = props.major_axis_length
    metrics["minor_axis_length"] = props.minor_axis_length
    metrics["orientation"] = props.orientation
    metrics["extent"] = props.extent
 
    return metrics
 
 
def main():
    group = zarr.open(ZARR_PATH, mode="r")
    available_ids = set(group.group_keys())
 
    df = pd.read_csv(CSV_PATH)
    df["image_id"] = df["image_id"].astype(str)
 
    metric_rows = []
    for img_id in df["image_id"]:
        if img_id not in available_ids:
            print(f"[WARN] {img_id} not found in zarr store — filling with NaN")
            metric_rows.append({})
            continue
 
        mask_arr = group[img_id][MASK_KEY]
        mask_3d = np.asarray(mask_arr) > 0
        max_proj = np.max(mask_3d, axis=Z_AXIS)
 
        print(f"Processing {img_id} — mask shape {mask_3d.shape} -> projection shape {max_proj.shape}")
        metric_rows.append(compute_morphology_2d(max_proj, pixel_size=PIXEL_SIZE_XY))
 
    metrics_df = pd.DataFrame(metric_rows)
    result_df = pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
 
    result_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved: {OUTPUT_CSV_PATH}")
 
 
if __name__ == "__main__":
    main()
 


# Read the CSV file
df = pd.read_csv(OUTPUT_CSV_PATH)

# Display the table
print(df)