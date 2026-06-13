"""Per-channel mean and standard deviation over the training split.

compute_stats() iterates the training CellDataset and returns
{'mean': [m0, m1], 'std': [s0, s1]} which is saved to outputs/stats.json.
load_stats() reads that file back.

Run once before training via scripts/compute_stats.py.
"""
