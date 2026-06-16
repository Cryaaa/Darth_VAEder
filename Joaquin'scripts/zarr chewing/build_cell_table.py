"""Build the per-cell lookup table (CSV) for the multinucleation.zarr store.

Iterates over EVERY cell in the zarr ``cell_index`` and writes one row per cell,
joining the microscopy metadata. This CSV is the table the data module reads;
the data loader then only ever deals in integer ``cell_idx`` (shuffle / split /
look up). Build it once per store.

Columns
-------
    cell_idx, replicate, condition, image_name, local_label_id,
    local_cell_index, microscope, magnification_x, pixel_size_um_per_px,
    scale_factor

Usage
-----
    python scripts/build_cell_table.py \
        --zarr /path/to/multinucleation.zarr \
        --image-metadata multinucleation_image_metadata.csv \
        --out outputs/cell_table.csv
"""

import argparse
import csv
from pathlib import Path

import zarr

BIAS_FIELDS = ("microscope", "magnification_x", "pixel_size_um_per_px", "scale_factor")


def build_cell_table(zarr_path, image_metadata_csv=None, bias_fields=BIAS_FIELDS):
    """Return (columns, rows) — one dict per cell — by iterating the cell_index."""
    root = zarr.open_group(str(zarr_path), mode="r")
    ci = root["cell_index"]
    ncells = ci["ncells_idx"][:]
    rep = ci["replicate"][:].astype(str)
    cond = ci["condition"][:].astype(str)
    img = ci["image_name"][:].astype(str)
    label = ci["local_label_id"][:]
    local = ci["local_cell_index"][:]

    # microscopy lookup keyed by (replicate, image_name)
    meta = {}
    if image_metadata_csv:
        with open(image_metadata_csv, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["sample"], Path(r["filename"]).stem)
                meta[key] = {b: r.get(b, "") for b in bias_fields}

    columns = ["cell_idx", "replicate", "condition", "image_name",
               "local_label_id", "local_cell_index", *bias_fields]
    rows = []
    for i in range(len(ncells)):
        row = {
            "cell_idx": int(ncells[i]),
            "replicate": rep[i],
            "condition": cond[i],
            "image_name": img[i],
            "local_label_id": int(label[i]),
            "local_cell_index": int(local[i]),
        }
        row.update(meta.get((rep[i], img[i]), {b: "" for b in bias_fields}))
        rows.append(row)
    return columns, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", required=True, help="Path to multinucleation.zarr")
    ap.add_argument("--image-metadata", default=None, help="Microscopy CSV (optional)")
    ap.add_argument("--out", required=True, help="Output cell-table CSV path")
    args = ap.parse_args()

    columns, rows = build_cell_table(args.zarr, args.image_metadata)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}: {len(rows)} cells, {len(columns)} columns")


if __name__ == "__main__":
    main()
