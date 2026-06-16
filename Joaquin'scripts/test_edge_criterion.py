"""Quick visual test of the edge-cell detection criterion.

Samples N cells randomly from the zarr, runs the cCellmask run-length criterion
at several thresholds, and saves a PNG grid so you can judge the cut-off visually.

Usage (on server):
    python "Joaquin'scripts/test_edge_criterion.py" \
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \
        --table outputs/cell_table.csv \
        --out   outputs/edge_test.png \
        --n     48 \
        --thresholds 5 10 15 20
"""

import argparse
import random
import numpy as np
import pandas as pd
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label as nd_label
from pathlib import Path


def max_run_on_border(cell_mask: np.ndarray) -> int:
    """Return the longest contiguous run of filled pixels along any of the 4 patch borders."""
    best = 0
    for strip in [cell_mask[0, :], cell_mask[-1, :], cell_mask[:, 0], cell_mask[:, -1]]:
        lbl, n = nd_label(strip > 0)
        if n:
            run = max(int((lbl == i).sum()) for i in range(1, n + 1))
            best = max(best, run)
    return best


def which_border(cell_mask: np.ndarray):
    """Return which borders have the longest run (for visualisation)."""
    borders = {
        "top":    cell_mask[0, :],
        "bottom": cell_mask[-1, :],
        "left":   cell_mask[:, 0],
        "right":  cell_mask[:, -1],
    }
    hits = []
    for name, strip in borders.items():
        lbl, n = nd_label(strip > 0)
        if n:
            run = max(int((lbl == i).sum()) for i in range(1, n + 1))
            if run > 0:
                hits.append((run, name))
    hits.sort(reverse=True)
    return hits


def load_sample(zarr_path: str, table_csv: str, n: int, seed: int = 42):
    """Randomly sample n cells and return their membrane image + cCellmask."""
    df  = pd.read_csv(table_csv)
    rng = random.Random(seed)
    idxs = rng.sample(range(len(df)), min(n, len(df)))

    z = zarr.open(zarr_path, mode="r")
    samples = []
    for idx in idxs:
        row  = df.iloc[idx]
        rep  = row["replicate"]
        cond = row["condition"]
        img  = row["image_name"]
        li   = int(row["local_cell_index"])

        pg = z[f"patches/{rep}/{cond}/{img}"]

        # membrane channel (ch0 of cnPatches if available, else cPatches)
        src = "cnPatches" if "cnPatches" in pg else "cPatches"
        mem = pg[src][li, :, :, 0]          # (256, 256) float32

        cmask = pg["cCellmask"][li]          # (256, 256) int32

        run = max_run_on_border(cmask)
        border_hits = which_border(cmask)

        samples.append({
            "idx":   idx,
            "rep":   rep,
            "cond":  cond,
            "img":   img,
            "mem":   mem,
            "cmask": cmask,
            "run":   run,
            "hits":  border_hits,
        })

    return samples


def plot_grid(samples, thresholds, out_path: str):
    n      = len(samples)
    n_thr  = len(thresholds)
    ncols  = 8
    nrows  = int(np.ceil(n / ncols))

    # One figure per threshold so you can compare side-by-side
    figs = []
    for thr in thresholds:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2, nrows * 2.2),
                                 squeeze=False)
        fig.suptitle(f"Edge criterion  |  max run-length threshold = {thr} px\n"
                     f"RED = flagged as cropped  |  GREEN = kept",
                     fontsize=11, y=1.01)

        for ax in axes.flat:
            ax.axis("off")

        for i, s in enumerate(samples):
            r, c = divmod(i, ncols)
            ax = axes[r][c]

            mem   = s["mem"]
            cmask = s["cmask"]
            run   = s["run"]
            cropped = run >= thr

            # normalise membrane for display
            lo, hi = np.percentile(mem[cmask > 0], [1, 99]) if cmask.any() else (mem.min(), mem.max())
            mem_n  = np.clip((mem - lo) / (hi - lo + 1e-6), 0, 1)

            # RGB: grey membrane
            rgb = np.stack([mem_n] * 3, axis=-1)

            # overlay cCellmask edge in colour
            border_overlay = np.zeros_like(cmask, dtype=bool)
            border_overlay[0, :]  = cmask[0, :] > 0
            border_overlay[-1, :] = cmask[-1, :] > 0
            border_overlay[:, 0]  = cmask[:, 0] > 0
            border_overlay[:, -1] = cmask[:, -1] > 0

            color = [1, 0.1, 0.1] if cropped else [0.1, 0.9, 0.1]
            rgb[border_overlay] = color

            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(
                f"run={run}px\n{'CROPPED' if cropped else 'ok'}",
                fontsize=7,
                color="red" if cropped else "green",
                pad=2,
            )
            ax.axis("off")

        fig.tight_layout()
        figs.append((thr, fig))

    # Save all thresholds into one PNG (stacked vertically via individual files)
    base = Path(out_path)
    saved = []
    for thr, fig in figs:
        p = base.with_name(base.stem + f"_thr{thr}" + base.suffix)
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))
        print(f"  saved: {p}")
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",       required=True)
    p.add_argument("--table",      required=True)
    p.add_argument("--out",        default="outputs/edge_test.png")
    p.add_argument("--n",          type=int,   default=48, help="Number of cells to sample")
    p.add_argument("--thresholds", type=int,   nargs="+", default=[5, 10, 15, 20],
                   help="Run-length thresholds to test (produces one PNG per threshold)")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    print(f"Loading {args.n} random cells …")
    samples = load_sample(args.zarr, args.table, args.n, args.seed)
    runs = [s["run"] for s in samples]
    print(f"  max run lengths: min={min(runs)}  median={int(np.median(runs))}  max={max(runs)}")
    print(f"  cells with run > 0:  {sum(r > 0 for r in runs)}")
    for thr in args.thresholds:
        n_crop = sum(r >= thr for r in runs)
        print(f"  threshold={thr:>3}px  →  {n_crop}/{args.n} flagged as cropped ({100*n_crop/args.n:.0f}%)")

    print("Plotting …")
    plot_grid(samples, args.thresholds, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
