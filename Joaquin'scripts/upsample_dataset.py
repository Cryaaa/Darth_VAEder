"""
Upsample all images and paired masks (cp_masks.png → .tiff, NucleiSeg.tif)
to a common physical resolution using per-image scale_factors from metadata.

Reference resolution: ID19 @ 20X EVOS = 0.448531 µm/px (scale_factor=1.0).
Images      → bilinear interpolation (order=1), preserves uint8/uint16 range.
Label masks → nearest-neighbour (order=0), preserves integer cell IDs.

Output images are always (2, H, W) = [membrane, nuclei] channel order.
Brightfield channels are dropped where present.

Channel mapping per sample/condition:
  ID18  all        : memb=ch0, nucl=ch1
  ID19  all        : memb=ch0, nucl=ch1
  ID23  all        : memb=ch1, nucl=ch0
  N2    CTRL/MAT   : memb=ch1, nucl=ch0
  N2    CMs25d     : memb=ch1, nucl=ch0  (ch2=brightfield → drop)
  N3    CMs25d     : memb=ch1, nucl=ch0
  N3    CTRL/MAT   : memb=ch0, nucl=ch1  (ch2=brightfield → drop)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.transform import resize
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ── channel configuration ─────────────────────────────────────────────────────

def get_channel_config(sample: str, condition: str):
    """
    Return (membrane_idx, nuclei_idx) after any brightfield channel is removed.
    Also returns drop_last=True when a trailing brightfield channel must be dropped.
    """
    s = sample.upper()
    c = condition.upper()

    if s in ("ID18", "ID19"):
        return dict(memb=0, nucl=1, drop_last=False)

    if s == "ID23":
        return dict(memb=1, nucl=0, drop_last=False)

    if s == "N2":
        drop = (c == "CMS25D")
        return dict(memb=1, nucl=0, drop_last=drop)

    if s == "N3":
        if c == "CMS25D":
            return dict(memb=1, nucl=0, drop_last=False)
        else:  # CTRL / MATURE
            return dict(memb=0, nucl=1, drop_last=True)

    raise ValueError(f"Unknown sample: {sample!r}")


def select_channels(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    From a (C, H, W) array, drop brightfield if needed, then reorder to
    [membrane, nuclei] → output shape (2, H, W).
    """
    if cfg["drop_last"]:
        img = img[:-1]          # remove last channel (brightfield)
    memb = img[cfg["memb"]]
    nucl = img[cfg["nucl"]]
    return np.stack([memb, nucl], axis=0)


# ── resize helpers ────────────────────────────────────────────────────────────

def upsample_image(arr: np.ndarray, scale: float) -> np.ndarray:
    """Bilinear upsample a (C,H,W) image, preserving dtype range."""
    if scale == 1.0:
        return arr
    orig_dtype = arr.dtype
    C, H, W = arr.shape
    new_H = round(H * scale)
    new_W = round(W * scale)
    out = np.stack(
        [resize(arr[c].astype(np.float64), (new_H, new_W),
                order=1, mode="reflect", anti_aliasing=True,
                preserve_range=True)
         for c in range(C)],
        axis=0,
    )
    max_val = np.iinfo(orig_dtype).max if np.issubdtype(orig_dtype, np.integer) else 1.0
    return np.clip(np.round(out), 0, max_val).astype(orig_dtype)


def upsample_mask(arr: np.ndarray, scale: float) -> np.ndarray:
    """Nearest-neighbour upsample a (H,W) label mask, preserving integer IDs."""
    if scale == 1.0:
        return arr
    orig_dtype = arr.dtype
    H, W = arr.shape
    new_H = round(H * scale)
    new_W = round(W * scale)
    out = resize(arr.astype(np.float64), (new_H, new_W),
                 order=0, mode="edge", anti_aliasing=False,
                 preserve_range=True)
    return np.round(out).astype(orig_dtype)


def load_cp_mask(path: Path) -> np.ndarray:
    """Load a 16-bit PNG cp_mask into a uint16 numpy array."""
    arr = np.array(Image.open(path))
    return arr.astype(np.uint16)


# ── QC visualisation ──────────────────────────────────────────────────────────

def label_to_rgba(mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    from matplotlib.cm import tab20
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for label in np.unique(mask[mask > 0]):
        color = tab20(int(label) % 20)
        region = mask == label
        rgba[region, :3] = color[:3]
        rgba[region, 3] = alpha
    return rgba


def make_qc_plots(processed: list, out_dir: Path, n_examples: int = 8):
    """4-column QC: membrane | nuclei | cp_masks on membrane | NucleiSeg on nuclei."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(processed), size=min(n_examples, len(processed)), replace=False)
    examples = [processed[i] for i in sorted(indices)]

    fig, axes = plt.subplots(len(examples), 4, figsize=(22, 5 * len(examples)))
    if len(examples) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Membrane (ch0)", "Nuclei (ch1)", "cp_masks on membrane", "NucleiSeg on nuclei"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row_idx, ex in enumerate(examples):
        img, cp, ns = ex["img"], ex["cp"], ex["ns"]
        memb, nucl = img[0], img[1]
        label = f"{ex['sample']}/{ex['stem']}\nscale×{ex['scale']:.2f}"

        def pct(ch):
            v0, v1 = np.percentile(ch, [1, 99])
            return dict(vmin=v0, vmax=v1)

        axes[row_idx, 0].imshow(memb, cmap="Reds_r",  **pct(memb))
        axes[row_idx, 0].set_ylabel(label, fontsize=8)

        axes[row_idx, 1].imshow(nucl, cmap="Blues_r", **pct(nucl))

        axes[row_idx, 2].imshow(memb, cmap="gray", **pct(memb))
        axes[row_idx, 2].imshow(label_to_rgba(cp), interpolation="nearest")

        axes[row_idx, 3].imshow(nucl, cmap="gray", **pct(nucl))
        axes[row_idx, 3].imshow(label_to_rgba(ns, alpha=0.5), interpolation="nearest")

        for ax in axes[row_idx]:
            ax.axis("off")

    fig.suptitle("Upsampling QC — [membrane | nuclei] channel order", fontsize=14, y=1.01)
    fig.tight_layout()
    qc_path = out_dir / "qc_upsample_overview.png"
    fig.savefig(qc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nQC figure saved → {qc_path}")
    return qc_path


# ── main processing ───────────────────────────────────────────────────────────

def process_dataset(data_dir: Path, out_dir: Path, meta_path: Path):
    meta = pd.read_csv(meta_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    skipped = []

    for _, row in meta.iterrows():
        sample    = row["sample"]
        fname     = row["filename"]
        condition = row["condition"]
        scale     = float(row["scale_factor"])
        stem      = Path(fname).stem

        cfg     = get_channel_config(sample, condition)
        src_dir = data_dir / sample
        dst_dir = out_dir / sample
        dst_dir.mkdir(parents=True, exist_ok=True)

        img_src = src_dir / fname
        cp_src  = src_dir / f"{stem}_cp_masks.png"
        ns_src  = src_dir / f"{stem}_NucleiSeg.tif"

        # ── image: drop BF if needed, reorder → [memb, nucl], upsample ──────
        if img_src.exists():
            raw  = tifffile.imread(img_src)          # (C, H, W)
            img  = select_channels(raw, cfg)          # (2, H, W) [memb, nucl]
            img_up = upsample_image(img, scale)
            tifffile.imwrite(dst_dir / fname, img_up,
                             photometric="minisblack", compression="deflate")
            shape_str = f"{raw.shape} → {img_up.shape}"
        else:
            print(f"  [WARN] missing image: {img_src.relative_to(data_dir)}")
            img_up = None
            shape_str = ""

        # ── cp_masks.png → cp_masks.tiff ────────────────────────────────────
        cp_dst = dst_dir / f"{stem}_cp_masks.tiff"
        if cp_src.exists():
            cp    = load_cp_mask(cp_src)
            cp_up = upsample_mask(cp, scale)
            tifffile.imwrite(cp_dst, cp_up, compression="deflate")
        else:
            print(f"  [WARN] missing cp_mask: {cp_src.relative_to(data_dir)}")
            cp_up = None

        # ── NucleiSeg.tif ────────────────────────────────────────────────────
        ns_dst = dst_dir / f"{stem}_NucleiSeg.tiff"
        if ns_src.exists():
            ns    = tifffile.imread(ns_src)
            ns_up = upsample_mask(ns, scale)
            tifffile.imwrite(ns_dst, ns_up, compression="deflate")
        else:
            print(f"  [WARN] missing NucleiSeg: {ns_src.relative_to(data_dir)}")
            ns_up = None

        print(f"  ✓  {sample}/{fname}  [{condition}]  scale={scale:.3f}  {shape_str}"
              f"  cfg=memb{cfg['memb']}/nucl{cfg['nucl']}"
              + ("  drop_last" if cfg["drop_last"] else ""))

        if img_up is not None and cp_up is not None and ns_up is not None:
            processed.append({
                "sample": sample, "stem": stem,
                "img": img_up, "cp": cp_up, "ns": ns_up,
                "scale": scale,
            })
        else:
            skipped.append(f"{sample}/{stem}")

    return processed, skipped


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE      = Path("/Users/joaco/Documents/Janelia/Multinucleation Big")
    DATA_DIR  = BASE / "data"
    OUT_DIR   = BASE / "data_upsampled"
    META_PATH = BASE / "multinucleation_image_metadata.csv"

    print(f"Source : {DATA_DIR}")
    print(f"Output : {OUT_DIR}")
    print(f"Meta   : {META_PATH}\n")

    processed, skipped = process_dataset(DATA_DIR, OUT_DIR, META_PATH)

    print(f"\n── Summary ──────────────────────────────")
    print(f"  Processed : {len(processed)}")
    print(f"  Skipped   : {len(skipped)}")
    for s in skipped:
        print(f"    • {s}")

    make_qc_plots(processed, OUT_DIR, n_examples=8)
    print("\nDone.")
