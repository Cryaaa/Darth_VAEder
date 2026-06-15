"""
migrate_to_zarr.py

Converts data_upsampled/ TIFFs into a single Zarr store with two layers:

  Layer 1 — full images
    root.zarr/images/<replicate>/<condition>/<image_name>/
      image       (2, H, W) float32   raw multichannel [membrane, nuclei]
      nucleiseg   (H, W)    int32     nuclei instance labels
      cpmask      (H, W)    int32     CellPose cell instance labels

  Layer 2 — per-cell patches (256×256)
    root.zarr/patches/<replicate>/<condition>/<image_name>/
      cPatches    (n, 256, 256, 2)  isolated cell crop, raw float32
      cCellmask   (n, 256, 256)     binary mask of target cell only
      cNucmask    (n, 256, 256)     nucleus labels inside this cell
      bbPatches   (n, 256, 256, 2)  context window centred on cell
      bbCellmask  (n, 256, 256)     all cell labels in the context window
      bbNucmask   (n, 256, 256)     all nucleus labels in context window

Global ncells index stored at root as JSON + Zarr arrays.

Validity filter: cells whose padded bbox (tight + 5px) exceeds 256 in either
axis are discarded.  bbPatches uses centroid-centred 256×256 windows with
clamp-and-shift at image borders.

Architecture
------------
Parallelised with multiprocessing.Pool (spawn-safe, all workers top-level):
  Phase 1  scan            (main, cheap)
  Phase 2  enumerate cells (pool — read every cpmask once, in parallel)
  Phase 3a preallocate     (main — create empty zarr structure at final shapes)
  Phase 3b fill            (pool — one image per task, writes its own groups)
  Phase 4  validate        (main)

Concurrency is race-free because chunk == 1 cell: every cell is its own chunk
file, and each worker owns a disjoint set of (image, patch) groups, so no two
workers ever write the same file.
"""

import os
import json
import re
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import tifffile
import zarr
import numcodecs
from pathlib import Path
from datetime import datetime
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

_STR_CODEC = numcodecs.VLenUTF8()

# ── constants ─────────────────────────────────────────────────────────────────

PATCH_SIZE = 256
PADDING    = 5
BASE       = Path("/Users/joaco/Documents/Janelia/Multinucleation Big")
SRC_ROOT   = BASE / "data_upsampled"
ZARR_PATH  = BASE / "multinucleation.zarr"

# Worker count: never exceed 80% of logical cores.
N_WORKERS  = max(1, int((os.cpu_count() or 1) * 0.8))


def _dilate_mask(binary: np.ndarray, radius: int) -> np.ndarray:
    """Morphological dilation with a disk SE of given radius."""
    if radius == 0:
        return binary.astype(bool)
    struct = iterate_structure(generate_binary_structure(2, 1), radius)
    return binary_dilation(binary, structure=struct).astype(bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Scan
# ═══════════════════════════════════════════════════════════════════════════════

def scan_dataset(src_root: Path) -> pd.DataFrame:
    """
    Walk data_upsampled/, parse filenames, return a DataFrame with one row
    per image and sibling paths to the three required files.
    """
    records = []
    cond_re = re.compile(r"(CTRL|MATURE|CMs25d)")

    for cp_path in sorted(src_root.rglob("*_cp_masks.tiff")):
        replicate = cp_path.parent.name
        stem      = cp_path.name.replace("_cp_masks.tiff", "")

        m = cond_re.search(stem)
        if not m:
            print(f"  [WARN] can't parse condition: {cp_path}")
            continue
        condition = m.group(1)

        img_path = cp_path.parent / f"{stem}.tif"
        ns_path  = cp_path.parent / f"{stem}_NucleiSeg.tiff"

        missing = [p for p in [img_path, ns_path, cp_path] if not p.exists()]
        if missing:
            print(f"  [WARN] missing files: {[str(p) for p in missing]}")
            continue

        records.append(dict(
            replicate   = replicate,
            condition   = condition,
            image_name  = stem,
            image_path  = str(img_path),
            nucleiseg_path = str(ns_path),
            cpmask_path = str(cp_path),
        ))

    df = pd.DataFrame(records)
    print(f"Scan found {len(df)} images across "
          f"{df['replicate'].nunique()} replicates, "
          f"{df['condition'].nunique()} conditions")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Enumerate: build global ncells index (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

def _tight_bbox(binary: np.ndarray):
    """(r0, r1_excl, c0, c1_excl) of True pixels."""
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(r1) + 1, int(c0), int(c1) + 1


def _padded_size(r0, r1, c0, c1) -> tuple[int, int]:
    """Height and width of the padded (unclamped) bbox."""
    return (r1 - r0 + 2 * PADDING), (c1 - c0 + 2 * PADDING)


def _scan_image_cells(cpmask_path: str) -> dict:
    """
    Worker: read one cpmask, apply the validity filter, return the ordered list
    of valid label ids plus shape and discard count.  No global state touched.
    """
    cp = tifffile.imread(cpmask_path).astype(np.int32)
    H, W = cp.shape
    labels = np.unique(cp)
    labels = labels[labels != 0]

    valid_labels = []
    n_discarded  = 0
    for label in labels:
        binary = cp == label
        if binary.sum() < 10:                       # artefact
            n_discarded += 1
            continue
        r0, r1, c0, c1 = _tight_bbox(binary)
        ph, pw = _padded_size(r0, r1, c0, c1)
        if ph > PATCH_SIZE or pw > PATCH_SIZE:       # too big for the window
            n_discarded += 1
            continue
        valid_labels.append(int(label))

    return dict(H=H, W=W, valid_labels=valid_labels, n_discarded=n_discarded)


def build_cell_index(df: pd.DataFrame, n_workers: int) -> tuple[list[dict], dict]:
    """
    Read every cpmask in parallel, then assign monotonic global ncells_idx in
    deterministic df order so the index stays contiguous 0..N-1.

    Returns
    -------
    cell_records : list[dict]   one entry per valid cell (global index)
    per_image    : dict         keyed by (replicate, condition, image_name)
    """
    paths = df["cpmask_path"].tolist()
    with mp.Pool(processes=n_workers, maxtasksperchild=4) as pool:
        results = pool.map(_scan_image_cells, paths)   # order preserved

    cell_records = []
    per_image    = {}
    global_idx   = 0

    for (_, row), res in zip(df.iterrows(), results):
        valid_labels = res["valid_labels"]
        ncells_start = global_idx

        for local_idx, label in enumerate(valid_labels):
            cell_records.append(dict(
                ncells_idx                    = global_idx,
                replicate                     = row["replicate"],
                condition                     = row["condition"],
                image_name                    = row["image_name"],
                local_label_id                = int(label),
                local_cell_index_within_image = local_idx,
            ))
            global_idx += 1

        key = (row["replicate"], row["condition"], row["image_name"])
        per_image[key] = dict(
            ncells_start = ncells_start,
            ncells_end   = global_idx,              # exclusive
            n_valid      = len(valid_labels),
            n_discarded  = res["n_discarded"],
            label_ids    = valid_labels,
            H            = res["H"],
            W            = res["W"],
        )

        print(f"  {row['replicate']}/{row['image_name']}: "
              f"valid={len(valid_labels)}, discarded={res['n_discarded']}")

    print(f"\nTotal valid cells (ncells): {global_idx}")
    return cell_records, per_image


# ═══════════════════════════════════════════════════════════════════════════════
# Patching helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _centroid_window(binary: np.ndarray, H: int, W: int) -> tuple[int, int, int, int]:
    """
    PATCH_SIZE × PATCH_SIZE window centred on the centroid of the binary mask.
    Clamp-and-shift keeps the window fully inside the image while keeping it
    exactly PATCH_SIZE in both axes.
    Returns (r0, r1, c0, c1) — the shared crop region for ALL six patch arrays.
    """
    P = PATCH_SIZE
    coords = np.argwhere(binary)
    cr = int(round(float(coords[:, 0].mean())))
    cc = int(round(float(coords[:, 1].mean())))

    r0 = cr - P // 2;  r1 = r0 + P
    c0 = cc - P // 2;  c1 = c0 + P

    if r0 < 0:    r0, r1 = 0, P
    elif r1 > H:  r0, r1 = H - P, H
    if c0 < 0:    c0, c1 = 0, P
    elif c1 > W:  c0, c1 = W - P, W

    return r0, r1, c0, c1


def _window_dilated(binary: np.ndarray, r0, r1, c0, c1, H, W) -> np.ndarray:
    """
    Dilated cell mask restricted to the [r0:r1, c0:c1] window, computed without
    dilating the full image.  We crop the binary mask to the window expanded by
    PADDING on every side (clamped to the image), dilate that small region, then
    slice back to the window.  Result is bit-identical to dilating the full mask
    and cropping, but operates on ~(P+2·PAD)² pixels instead of H·W.
    """
    er0 = max(0, r0 - PADDING); er1 = min(H, r1 + PADDING)
    ec0 = max(0, c0 - PADDING); ec1 = min(W, c1 + PADDING)
    dil = _dilate_mask(binary[er0:er1, ec0:ec1], PADDING)
    return dil[r0 - er0: r0 - er0 + (r1 - r0),
               c0 - ec0: c0 - ec0 + (c1 - c0)]


def make_all_patches(label:     int,
                     image:     np.ndarray,   # (2, H, W) float32
                     cpmask:    np.ndarray,   # (H, W) int32
                     nucleiseg: np.ndarray,   # (H, W) int32
                     ) -> tuple[np.ndarray, ...]:
    """
    Compute all six patch arrays for one cell, guaranteed to be spatially
    registered by sharing a single centroid-centred 256×256 window.

    cPatches   (P,P,2)  masked image channels (dilated mask, cell isolated)
    cCellmask  (P,P)    original (non-dilated) binary cell mask
    cNucmask   (P,P)    nucleus instance IDs within the exact cell boundary
    bbPatches  (P,P,2)  raw image (all context preserved)
    bbCellmask (P,P)    full cpmask in the window (all cell labels)
    bbNucmask  (P,P)    full nucleiseg in the window (all nucleus labels)
    """
    _, H, W = image.shape
    binary  = (cpmask == label)

    # shared window — one centroid, one crop region for everything
    r0, r1, c0, c1 = _centroid_window(binary, H, W)

    # window-local crops (no full-image dilation)
    binary_win = binary[r0:r1, c0:c1]
    dilated_win = _window_dilated(binary, r0, r1, c0, c1, H, W)

    memb_win = image[0, r0:r1, c0:c1]
    nucl_win = image[1, r0:r1, c0:c1]

    # ── c* : isolated cell (masked image, same window) ────────────────────────
    cPatch    = np.stack([memb_win * dilated_win,
                          nucl_win * dilated_win], axis=-1).astype(np.float32)
    cCellmask = binary_win.astype(np.int32)                                 # exact cell shape
    cNucmask  = (nucleiseg[r0:r1, c0:c1] * binary_win).astype(np.int32)     # nuclei inside cell

    # ── bb* : context window (raw image, same window) ────────────────────────
    bbPatch    = image[:, r0:r1, c0:c1].transpose(1, 2, 0).astype(np.float32)
    bbCellmask = cpmask[r0:r1, c0:c1].astype(np.int32)                      # all cell labels
    bbNucmask  = nucleiseg[r0:r1, c0:c1].astype(np.int32)                   # all nucleus labels

    return cPatch, cCellmask, cNucmask, bbPatch, bbCellmask, bbNucmask


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3a — Preallocate empty zarr structure (main process)
# ═══════════════════════════════════════════════════════════════════════════════

def preallocate(df: pd.DataFrame,
                cell_records: list[dict],
                per_image: dict,
                zarr_path: Path,
                src_root: Path):
    """
    Create the full empty store: root attrs, global cell_index arrays, and every
    image/patch group with arrays at their final shapes.  No pixel data written
    here — workers fill the chunks in parallel afterwards.
    """
    root = zarr.open_group(str(zarr_path), mode="a")

    root.attrs.update(dict(
        total_ncells = len(cell_records),
        patch_size   = PATCH_SIZE,
        padding      = PADDING,
        date_created = datetime.now().isoformat(timespec="seconds"),
        source_root  = str(src_root),
    ))

    # ── global cell index as zarr arrays + JSON ───────────────────────────────
    ci = root.require_group("cell_index")
    N  = len(cell_records)
    ci.array("ncells_idx",       np.array([r["ncells_idx"]     for r in cell_records], np.int32), overwrite=True)
    ci.array("local_label_id",   np.array([r["local_label_id"] for r in cell_records], np.int32), overwrite=True)
    ci.array("local_cell_index", np.array([r["local_cell_index_within_image"] for r in cell_records], np.int32), overwrite=True)
    ci.array("replicate",  np.array([r["replicate"]  for r in cell_records], dtype=object), dtype=object, object_codec=_STR_CODEC, overwrite=True)
    ci.array("condition",  np.array([r["condition"]  for r in cell_records], dtype=object), dtype=object, object_codec=_STR_CODEC, overwrite=True)
    ci.array("image_name", np.array([r["image_name"] for r in cell_records], dtype=object), dtype=object, object_codec=_STR_CODEC, overwrite=True)
    print(f"Wrote cell_index ({N} entries)")

    with open(zarr_path / "cell_index.json", "w") as f:
        json.dump(cell_records, f, indent=2)

    # ── per-image empty groups/arrays ─────────────────────────────────────────
    P, C = PATCH_SIZE, 2
    for _, row in df.iterrows():
        rep, cond, iname = row["replicate"], row["condition"], row["image_name"]
        meta = per_image[(rep, cond, iname)]
        H, W = meta["H"], meta["W"]
        n    = meta["n_valid"]

        ig = root.require_group(f"images/{rep}/{cond}/{iname}")
        ig.empty("image",     shape=(2, H, W), chunks=(1, H, W), dtype="f4", overwrite=True)
        ig.empty("nucleiseg", shape=(H, W),    chunks=(H, W),    dtype="i4", overwrite=True)
        ig.empty("cpmask",    shape=(H, W),    chunks=(H, W),    dtype="i4", overwrite=True)
        ig.attrs.update(dict(
            replicate=rep, condition=cond, image_name=iname,
            source_path=row["image_path"], _complete=False,
        ))

        pg = root.require_group(f"patches/{rep}/{cond}/{iname}")
        if n > 0:
            pg.empty("cPatches",  shape=(n, P, P, C), chunks=(1, P, P, C), dtype="f4", overwrite=True)
            pg.empty("cCellmask", shape=(n, P, P),    chunks=(1, P, P),    dtype="i4", overwrite=True)
            pg.empty("cNucmask",  shape=(n, P, P),    chunks=(1, P, P),    dtype="i4", overwrite=True)
            pg.empty("bbPatches", shape=(n, P, P, C), chunks=(1, P, P, C), dtype="f4", overwrite=True)
            pg.empty("bbCellmask",shape=(n, P, P),    chunks=(1, P, P),    dtype="i4", overwrite=True)
            pg.empty("bbNucmask", shape=(n, P, P),    chunks=(1, P, P),    dtype="i4", overwrite=True)
        pg.attrs.update(dict(
            replicate=rep, condition=cond, image_name=iname,
            ncells_start=meta["ncells_start"], ncells_end=meta["ncells_end"],
            n_valid_cells=meta["n_valid"], n_discarded_cells=meta["n_discarded"],
            label_ids=meta["label_ids"],
        ))

    print(f"Preallocated {len(df)} image/patch groups")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3b — Fill (parallel worker, one image per task)
# ═══════════════════════════════════════════════════════════════════════════════

def _fill_image(task: tuple, zarr_path: str) -> str:
    """
    Worker: load one image's TIFFs, write Layer 1 arrays and all per-cell Layer 2
    patches into the preallocated store.  Idempotent / resumable: skips images
    already flagged _complete.  Owns disjoint groups → race-free.
    """
    row, meta = task
    rep, cond, iname = row["replicate"], row["condition"], row["image_name"]

    root = zarr.open_group(zarr_path, mode="a")
    ig   = root[f"images/{rep}/{cond}/{iname}"]

    if ig.attrs.get("_complete"):
        return f"  [SKIP] {rep}/{iname}"

    image     = tifffile.imread(row["image_path"]).astype(np.float32)   # (2,H,W)
    nucleiseg = tifffile.imread(row["nucleiseg_path"]).astype(np.int32)
    cpmask    = tifffile.imread(row["cpmask_path"]).astype(np.int32)

    # Layer 1
    ig["image"][:]     = image
    ig["nucleiseg"][:] = nucleiseg
    ig["cpmask"][:]    = cpmask

    # Layer 2
    n = meta["n_valid"]
    if n > 0:
        pg = root[f"patches/{rep}/{cond}/{iname}"]
        cP, cCM, cNM = pg["cPatches"], pg["cCellmask"], pg["cNucmask"]
        bbP, bbCM, bbNM = pg["bbPatches"], pg["bbCellmask"], pg["bbNucmask"]
        for i, label in enumerate(meta["label_ids"]):
            cP[i], cCM[i], cNM[i], bbP[i], bbCM[i], bbNM[i] = \
                make_all_patches(label, image, cpmask, nucleiseg)

    ig.attrs["_complete"] = True
    return f"  ✓  {rep}/{iname}  ncells={n}"


def fill_parallel(df: pd.DataFrame, per_image: dict,
                  zarr_path: Path, n_workers: int):
    tasks = [(row.to_dict(), per_image[(row["replicate"], row["condition"], row["image_name"])])
             for _, row in df.iterrows()]
    worker = partial(_fill_image, zarr_path=str(zarr_path))

    print(f"Filling {len(tasks)} images with {n_workers} workers ...")
    with mp.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(worker, tasks):
            print(msg, flush=True)

    print(f"\nMigration complete → {zarr_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate(zarr_path: Path, cell_records: list[dict], per_image: dict):
    print("\n── Validation ───────────────────────────────────")
    root = zarr.open_group(str(zarr_path), mode="r")
    errors = []

    total = root.attrs["total_ncells"]
    if len(cell_records) != total:
        errors.append(f"total_ncells mismatch: {len(cell_records)} vs {total}")

    idxs = [r["ncells_idx"] for r in cell_records]
    if idxs != list(range(total)):
        errors.append("ncells_idx not contiguous 0..total_ncells-1")

    patch_total = 0
    for (rep, cond, iname), meta in per_image.items():
        # every image must have been flagged complete
        ig_path = f"images/{rep}/{cond}/{iname}"
        if ig_path not in root or not root[ig_path].attrs.get("_complete"):
            errors.append(f"Image not complete: {ig_path}")

        grp_path = f"patches/{rep}/{cond}/{iname}"
        if grp_path not in root:
            errors.append(f"Missing patch group: {grp_path}")
            continue
        pg = root[grp_path]
        n  = meta["n_valid"]
        if n == 0:
            continue
        for arr_name, expected_ndim in [("cPatches",4),("cCellmask",3),("cNucmask",3),
                                         ("bbPatches",4),("bbCellmask",3),("bbNucmask",3)]:
            arr = pg[arr_name]
            if arr.shape[0] != n:
                errors.append(f"{grp_path}/{arr_name}: leading dim {arr.shape[0]} ≠ {n}")
            if arr.ndim != expected_ndim:
                errors.append(f"{grp_path}/{arr_name}: ndim {arr.ndim} ≠ {expected_ndim}")
            spatial = arr.shape[1:3]
            if spatial != (PATCH_SIZE, PATCH_SIZE):
                errors.append(f"{grp_path}/{arr_name}: spatial {spatial} ≠ ({PATCH_SIZE},{PATCH_SIZE})")
        patch_total += n

    if patch_total != total:
        errors.append(f"Sum of n_valid across images ({patch_total}) ≠ total_ncells ({total})")

    for (rep, cond, iname), meta in per_image.items():
        lids = meta["label_ids"]
        if len(lids) != len(set(lids)):
            errors.append(f"{rep}/{iname}: duplicate label_ids")

    if errors:
        print("ERRORS:")
        for e in errors: print(f"  ✗ {e}")
    else:
        print(f"  ✓ total_ncells = {total}")
        print(f"  ✓ ncells_idx contiguous")
        print(f"  ✓ all images flagged complete")
        print(f"  ✓ all patch arrays shape-correct")
        print(f"  ✓ no duplicate label_ids")
        print("All checks passed.")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # safe with numpy/scipy/zarr on macOS

    print(f"Source  : {SRC_ROOT}")
    print(f"Output  : {ZARR_PATH}")
    print(f"Workers : {N_WORKERS} (of {os.cpu_count()} logical cores)\n")

    print("── Phase 1: Scan ─────────────────────────────────")
    df = scan_dataset(SRC_ROOT)

    print("\n── Phase 2: Enumerate (parallel) ─────────────────")
    cell_records, per_image = build_cell_index(df, N_WORKERS)

    print("\n── Phase 3a: Preallocate ─────────────────────────")
    preallocate(df, cell_records, per_image, ZARR_PATH, SRC_ROOT)

    print("\n── Phase 3b: Fill (parallel) ─────────────────────")
    fill_parallel(df, per_image, ZARR_PATH, N_WORKERS)

    print("\n── Phase 4: Validate ─────────────────────────────")
    validate(ZARR_PATH, cell_records, per_image)
