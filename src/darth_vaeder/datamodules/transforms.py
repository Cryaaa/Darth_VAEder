"""Image transforms for training and validation.

Training augmentations: random horizontal/vertical flip, random rotation.
Validation: no augmentation (normalisation handled in the dataset).
All transforms operate on (image, mask) pairs to keep them in sync.
"""
