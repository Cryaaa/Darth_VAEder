"""Entry point for VAE training.

Loads config, builds MultinucDataModule + LitVAE, and calls Trainer.fit().
Checkpoints saved to outputs/checkpoints/, logs to outputs/logs/.

Usage
-----
    python scripts/train.py --config configs/train.yaml
"""
