"""Generate a 3D rotating UMAP video from saved VAE embeddings.

Loads vae_v31.npz (Z + cell_idx), joins conditions from cell_table.csv,
runs UMAP with n_components=3, and renders a 360° rotating scatter as MP4.

Usage
-----
    python "Joaquin'scripts/make_umap3d_video.py" \
        --emb   outputs/embeddings/vae_v31.npz \
        --table outputs/cell_table.csv \
        --out   outputs/umap_explorer/umap3d_v31.mp4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PAL = {
    "CTRL":    "#4e9de0",
    "MATURE":  "#ff8c42",
    "CMs25d":  "#e040fb",
}
DEFAULT_PAL = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb",    default="/mnt/efs/dl_jrc/student_data/S-JS/repos/Darth_VAEder/Joaquin'scripts/outputs/embeddings/vae_v31.npz")
    p.add_argument("--table",  default="outputs/cell_table.csv")
    p.add_argument("--out",    default="outputs/umap_explorer/umap3d_v31.mp4")
    p.add_argument("--fps",    type=int,   default=30)
    p.add_argument("--n-rotations", type=int, default=2,
                   help="Number of full 360° rotations in the video")
    p.add_argument("--duration", type=float, default=12.0,
                   help="Total video duration in seconds")
    p.add_argument("--dpi",    type=int,   default=150)
    p.add_argument("--size",   type=int,   default=6,
                   help="Point size in scatter")
    p.add_argument("--alpha",  type=float, default=0.7)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist",    type=float, default=0.1)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── load embeddings ───────────────────────────────────────────────────────
    data = np.load(args.emb)
    Z        = data["Z"]          # (N, z_dim)
    cell_idx = data["cell_idx"]   # (N,)
    print(f"Loaded embeddings: {Z.shape}  cell_idx: {cell_idx.shape}")

    # ── join conditions ───────────────────────────────────────────────────────
    df = pd.read_csv(args.table).set_index("cell_idx")
    conditions = np.array([df.loc[ci, "condition"] for ci in cell_idx])
    unique_conds = list(dict.fromkeys(conditions))
    print(f"Conditions: {unique_conds}")

    # ── 3D UMAP ───────────────────────────────────────────────────────────────
    print("Running 3D UMAP …")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=3,
        random_state=42,
        verbose=True,
    )
    xyz = reducer.fit_transform(Z)   # (N, 3)
    print(f"UMAP done: {xyz.shape}")

    # ── colour map ────────────────────────────────────────────────────────────
    cmap = {}
    extra = iter(DEFAULT_PAL)
    for cond in unique_conds:
        cmap[cond] = PAL.get(cond, next(extra))

    # ── figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8), facecolor="#0d0d0d")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

    # scatter each condition as a separate artist so legend works
    scatters = {}
    for cond in unique_conds:
        mask = conditions == cond
        scatters[cond] = ax.scatter(
            xyz[mask, 0], xyz[mask, 1], xyz[mask, 2],
            c=cmap[cond], s=args.size, alpha=args.alpha,
            label=cond, depthshade=True, linewidths=0,
        )

    ax.set_title("VAE v31 — UMAP 3D", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.set_xlabel("UMAP 1", labelpad=6)
    ax.set_ylabel("UMAP 2", labelpad=6)
    ax.set_zlabel("UMAP 3", labelpad=6)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    legend = ax.legend(
        loc="upper left", framealpha=0.15, labelcolor="white",
        facecolor="#111", edgecolor="#444", fontsize=11,
        markerscale=3,
    )
    plt.tight_layout()

    # ── animation ─────────────────────────────────────────────────────────────
    n_frames   = int(args.fps * args.duration)
    deg_total  = 360.0 * args.n_rotations
    elev_start = 20.0
    elev_amp   = 10.0   # gentle up-down bob

    def update(frame):
        t     = frame / n_frames
        azim  = deg_total * t
        elev  = elev_start + elev_amp * np.sin(2 * np.pi * t * args.n_rotations)
        ax.view_init(elev=elev, azim=azim)
        return list(scatters.values())

    print(f"Rendering {n_frames} frames at {args.fps} fps …")
    writer = FFMpegWriter(fps=args.fps, bitrate=4000,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi=args.dpi):
        for i in range(n_frames):
            update(i)
            writer.grab_frame()
            if i % (args.fps * 2) == 0:
                print(f"  frame {i}/{n_frames}", end="\r")

    print(f"\nSaved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
