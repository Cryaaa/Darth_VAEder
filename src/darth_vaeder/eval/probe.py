"""Downstream evaluation on frozen embeddings.

linear_probe(): train a logistic regression on frozen latents to predict
condition (CTRL / MATURE / CMs25d) — measures how linearly separable the
learned representation is.

Also contains UMAP visualisation coloured by condition, replicate, and
predicted nucleus count.
"""
