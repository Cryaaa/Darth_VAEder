"""Extract latent embeddings for all cells using a trained checkpoint.

Loads LitVAE from checkpoint, runs the full dataset through the encoder,
and saves outputs/embeddings/embeddings.npz.

Usage
-----
    python scripts/extract_embeddings.py \
        --checkpoint outputs/checkpoints/best.ckpt \
        --zarr multinucleation.zarr \
        --splits outputs/splits.json \
        --out outputs/embeddings/embeddings.npz
"""
