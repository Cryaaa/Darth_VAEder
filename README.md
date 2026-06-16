# Multinucleation VAE — Local Notes

This folder is the development workspace for the Darth VAEder project applied to myotube multinucleation imaging data.

## What this is

A β-VAE trained on single-cell patches (256×256 px) from WGA-stained myotubes. Each cell contributes two image channels (membrane WGA + nuclear NLS) plus a crop mask. The model learns a compact 10-dimensional representation that we hope captures multinucleation state and morphology.

## Repo

GitHub: `Cryaaa/Darth_VAEder`, branch `JS_training`

## Quick start (server training)

```bash
ssh S-JS@3.17.63.55
conda activate darth-vaeder
cd /home/S-JS/Darth_VAEder
git pull origin JS_training

python "Joaquin'scripts/train.py" \
    --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
    --table outputs/cell_table.csv \
    --out   outputs \
    --epochs 50
```

## Data

| Dataset | Location | Channels | Replicates | Notes |
|---|---|---|---|---|
| WGA + NLS | zarr on server | membrane, nuclei | N1–N3 | 16,678 cells in zarr |
| Phalloidin + DAPI | `/Users/joaco/Documents/Janelia/Phalloidin and DAPI/` | actin, nuclei | IDM09–IDM24 | local TIFFs, not yet in zarr |

## Pipeline stages

1. **Raw TIFFs** → `migrate_to_zarr.py` → `multinucleation.zarr` (on server EFS)
2. **zarr enrichment** → `add_pCellmask.py` → adds `pCellmask` per cell group
3. **Training** → `train.py` → checkpoints + TensorBoard logs in `outputs/`

## See also

`CLAUDE.md` — detailed notes for Claude Code sessions (architecture, gotchas, decisions)
