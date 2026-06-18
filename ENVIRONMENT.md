# Environment setup

The pipeline has three dependency profiles, encoded as optional groups in
`pyproject.toml`:

| Group  | Stage                              | Adds                                  |
|--------|------------------------------------|---------------------------------------|
| (core) | VAE training                       | lightning, torch, numpy, pandas, zarr, matplotlib |
| `data` | raw images → `multinucleation.zarr`| tifffile, scipy, cellpose, numcodecs  |
| `eval` | latent-space analysis              | scikit-learn, umap-learn              |
| `dev`  | contributor tooling                | pre-commit, pytest, ruff              |

Core is always installed; pick the extras you need per machine.

> **Never install into conda `base`.** Always create a dedicated env so the
> pipeline is reproducible and base stays clean.

---

## Mac / data-prep / development

Builds the zarr, edits code, runs quick tests. Wants everything:

```bash
conda env create -f environment.yml      # creates env "darth-vaeder"
conda activate darth-vaeder
```

`environment.yml` runs `pip install -e .[data,eval,dev]` for you. (torch resolves
to the macOS / MPS wheel automatically.)

To rebuild from scratch:

```bash
conda env remove -n darth-vaeder
conda env create -f environment.yml
```

---

## Server (GPU training)

Training only — no `data` extras needed (the zarr is transferred in, not built
here). torch must match the server's CUDA driver, so install it explicitly
BEFORE the editable install.

```bash
# 1. isolated env
conda create -n darth-vaeder python=3.12 -y
conda activate darth-vaeder

# 2. torch for the server's CUDA (check `nvidia-smi` for the driver version)
#    example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. the package + eval extras (torch already satisfied, won't be overridden)
git clone https://github.com/Cryaaa/Darth_VAEder.git
cd Darth_VAEder && git checkout JoaquinProject
pip install -e ".[eval]"
```

Then point the data module at the transferred store:

```python
MultinucDataModule(data_path="/path/on/server/multinucleation.zarr", ...)
```

---

## Verify

```bash
python -c "import torch, lightning, zarr; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('lightning', lightning.__version__, 'zarr', zarr.__version__)"

# data env only:
python -c "import tifffile, scipy, cellpose; print('data deps OK')"

# import the package:
python -c "from darth_vaeder.datamodules import MultinucDataModule; print('package OK')"
```

`torch.cuda.is_available()` should be `True` on the server and `False` on the Mac
(use `torch.backends.mps.is_available()` for Apple-GPU there).
