# Example: a small PyTorch Lightning tutorial

[`lightning_tutorial.py`](lightning_tutorial.py) is a single, self-contained file
that teaches the three core pieces of PyTorch Lightning by training a tiny
autoencoder on made-up data (no downloads needed).

## What you'll learn

- **Dataset / DataLoader** — where the data comes from
- **LightningModule** — your model plus `training_step`, `validation_step`, and `configure_optimizers`
- **Trainer** — runs the training loop for you (and uses your GPU automatically)

## Run it

First install the project (see the top-level `CONTRIBUTING.md` for the full
conda setup):

```bash
pip install -e ".[dev]"
```

Then either run it straight through as a script:

```bash
python example/lightning_tutorial.py
```

…or open `lightning_tutorial.py` in VS Code or Jupyter and run the `# %%` cells
one at a time to follow along.
