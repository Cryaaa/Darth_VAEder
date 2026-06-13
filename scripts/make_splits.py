"""Assign cells to train / val / test splits by image.

Reads multinucleation.zarr cell_index and multinucleation_image_metadata.csv,
assigns each image to a split (stratified by condition), then maps all cells
from that image to the same split.

Saves outputs/splits.json: {'train': [...], 'val': [...], 'test': [...]}.

Usage
-----
    python scripts/make_splits.py --zarr multinucleation.zarr \
        --meta multinucleation_image_metadata.csv \
        --val-frac 0.15 --test-frac 0.15 --seed 42 \
        --out outputs/splits.json
"""
