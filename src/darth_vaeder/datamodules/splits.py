"""Train / val / test split assignment.

Splits are made at the IMAGE level (not the cell level) so that all cells from
the same field of view land in the same split — preventing data leakage.

Produces a dict: {'train': [ncells_idx, ...], 'val': [...], 'test': [...]}
that is saved to configs/splits.json and consumed by MultinucDataModule.
"""
