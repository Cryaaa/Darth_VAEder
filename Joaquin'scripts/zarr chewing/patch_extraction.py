"""
Cell patch extraction from upsampled multichannel images + segmentation masks.

Pipeline for each cell label in cp_masks
-----------------------------------------
1.  Extract the binary mask for that cell (irregular shape from CellPose).
2.  Morphologically dilate the mask by `padding` pixels (disk SE) →
    padded_mask is still irregularly shaped, just slightly expanded.
3.  Apply padded_mask to ALL channels:
      - membrane   : zeroed outside this cell's territory
      - nuclei     : zeroed outside → only nuclei belonging to this cell visible
      - nuclei_seg : zeroed outside → only nucleus labels within this cell
4.  Find the tight bounding box of padded_mask → rectangular crop region.
5.  Crop all three masked arrays with the same bbox → registered.
6.  Centre each crop in a (patch_size × patch_size) blank canvas.
    If the crop exceeds patch_size it is centre-cropped to fit.

Expected image layout
----------------------
image      : (2, H, W)  –  ch0 = membrane, ch1 = nuclei
cp_masks   : (H, W)     –  integer cell instance labels (CellPose)
nuclei_seg : (H, W)     –  integer nucleus instance labels

Expected dataloader batch keys
--------------------------------
  'image'      : torch.Tensor  (B, 2, H, W)
  'cp_masks'   : torch.Tensor  (B, H, W)
  'nuclei_seg' : torch.Tensor  (B, H, W)
  'filename'   : list[str]     length B  (optional)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure, iterate_structure


# ── data container ────────────────────────────────────────────────────────────

@dataclass
class CellPatch:
    """
    Three registered (patch_size × patch_size) arrays for one cell.

      membrane   : float32 [0,1]  masked to this cell's dilated territory
      nuclei     : float32 [0,1]  masked to this cell's dilated territory
      nuclei_seg : uint16         nucleus instance labels within this cell
    """
    membrane:    np.ndarray
    nuclei:      np.ndarray
    nuclei_seg:  np.ndarray

    cell_label:   int   = 0
    filename:     str   = ""
    bbox_padded:  tuple = field(default_factory=tuple)
    canvas_offset: tuple = field(default_factory=tuple)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dilate_mask(binary_mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Morphologically dilate a binary mask by `radius` pixels using a disk SE.
    Returns a bool array of the same shape.
    """
    if radius == 0:
        return binary_mask.astype(bool)
    # build a disk-shaped structuring element of the requested radius
    struct = iterate_structure(generate_binary_structure(2, 1), radius)
    return binary_dilation(binary_mask, structure=struct).astype(bool)


def _mask_bbox(binary_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bounding box (r0, r1_excl, c0, c1_excl) of True pixels."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(r1) + 1, int(c0), int(c1) + 1


def _place_in_canvas(crop: np.ndarray, patch_size: int,
                     fill: float | int = 0) -> tuple[np.ndarray, tuple[int, int]]:
    """Centre crop in a (patch_size, patch_size) canvas, centre-crop if too large."""
    P = patch_size
    H_c, W_c = crop.shape

    if H_c > P:
        s = (H_c - P) // 2
        crop = crop[s: s + P, :]
        H_c = P
    if W_c > P:
        s = (W_c - P) // 2
        crop = crop[:, s: s + P]
        W_c = P

    canvas = np.full((P, P), fill, dtype=crop.dtype)
    r_off = (P - H_c) // 2
    c_off = (P - W_c) // 2
    canvas[r_off: r_off + H_c, c_off: c_off + W_c] = crop
    return canvas, (r_off, c_off)


def _norm(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi == lo else (a - lo) / (hi - lo)


# ── core extraction ───────────────────────────────────────────────────────────

def extract_cell_patches(
    image:      np.ndarray,   # (2, H, W)
    cp_masks:   np.ndarray,   # (H, W) integer cell labels
    nuclei_seg: np.ndarray,   # (H, W) integer nucleus labels
    patch_size: int = 128,
    padding:    int = 5,
    filename:   str = "",
    min_pixels: int = 10,
) -> list[CellPatch]:
    """
    Generate one CellPatch per cell label in cp_masks.

    The same morphologically-dilated irregular mask is applied to membrane,
    nuclei, and nuclei_seg before cropping, so each patch contains only the
    signal belonging to this specific cell.
    """
    assert image.ndim == 3 and image.shape[0] == 2, \
        f"Expected image (2, H, W), got {image.shape}"
    assert cp_masks.ndim == nuclei_seg.ndim == 2

    membrane_ch = image[0]
    nuclei_ch   = image[1]

    patches: list[CellPatch] = []

    for label in np.unique(cp_masks):
        if label == 0:
            continue

        cell_mask = cp_masks == label
        if cell_mask.sum() < min_pixels:
            continue

        # 1. dilate the irregular mask by `padding` pixels
        padded_mask = _dilate_mask(cell_mask, padding)

        # 2. apply the same padded mask to every channel
        membrane_masked = membrane_ch * padded_mask
        nuclei_masked   = nuclei_ch   * padded_mask
        nuclei_seg_masked = nuclei_seg * padded_mask   # zeros nuclei outside this cell

        # 3. tight bbox of the padded mask → shared rectangular crop
        r0, r1, c0, c1 = _mask_bbox(padded_mask)

        membrane_crop = membrane_masked[r0:r1, c0:c1]
        nuclei_crop   = nuclei_masked[r0:r1, c0:c1]
        nseg_crop     = nuclei_seg_masked[r0:r1, c0:c1]

        # 4. centre each crop in the canvas (same offset → registered)
        memb_canvas,  (r_off, c_off) = _place_in_canvas(_norm(membrane_crop), patch_size, 0.0)
        nucl_canvas,  _              = _place_in_canvas(_norm(nuclei_crop),   patch_size, 0.0)
        nseg_canvas,  _              = _place_in_canvas(nseg_crop.astype(np.uint16), patch_size, 0)

        patches.append(CellPatch(
            membrane      = memb_canvas,
            nuclei        = nucl_canvas,
            nuclei_seg    = nseg_canvas,
            cell_label    = int(label),
            filename      = filename,
            bbox_padded   = (r0, r1, c0, c1),
            canvas_offset = (r_off, c_off),
        ))

    return patches


# ── dataloader wrapper ────────────────────────────────────────────────────────

def generate_patches_from_dataloader(
    dataloader,
    patch_size: int = 128,
    padding:    int = 5,
    min_pixels: int = 10,
) -> Iterator[CellPatch]:
    """
    Iterate a PyTorch / Lightning dataloader and yield one CellPatch per cell.
    Batch must contain: 'image', 'cp_masks', 'nuclei_seg', optionally 'filename'.
    """
    for batch in dataloader:
        images      = _to_numpy(batch["image"])
        cp_masks_b  = _to_numpy(batch["cp_masks"])
        nuclei_segs = _to_numpy(batch["nuclei_seg"])
        filenames   = batch.get("filename", [""] * images.shape[0])

        for i in range(images.shape[0]):
            fname = filenames[i] if isinstance(filenames[i], str) else str(filenames[i])
            yield from extract_cell_patches(
                image       = images[i],
                cp_masks    = cp_masks_b[i],
                nuclei_seg  = nuclei_segs[i],
                patch_size  = patch_size,
                padding     = padding,
                filename    = fname,
                min_pixels  = min_pixels,
            )


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ── convenience stacking ──────────────────────────────────────────────────────

def patches_to_arrays(patches: list[CellPatch]):
    """
    Stack patches into (N, P, P) arrays ready for a VAE.

    Returns
    -------
    membrane   : (N, P, P) float32
    nuclei     : (N, P, P) float32
    nuclei_seg : (N, P, P) uint16
    meta       : list[dict]
    """
    return (
        np.stack([p.membrane   for p in patches]),
        np.stack([p.nuclei     for p in patches]),
        np.stack([p.nuclei_seg for p in patches]),
        [{"cell_label":    p.cell_label,
          "filename":      p.filename,
          "bbox_padded":   p.bbox_padded,
          "canvas_offset": p.canvas_offset}
         for p in patches],
    )
