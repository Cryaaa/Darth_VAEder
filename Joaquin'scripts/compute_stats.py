"""Compute per-channel mean and std over the training split.

Reads the zarr store and splits.json, iterates training cells,
and saves outputs/stats.json.

Usage
-----
    python scripts/compute_stats.py --zarr multinucleation.zarr \
        --splits outputs/splits.json --out outputs/stats.json
"""
