#!/usr/bin/env python
"""Interactive centroid annotation for TIFF and CZI image stacks using napari.

Click in the ``centroids`` points layer to mark cell centroids, then use the
dock-widget buttons to navigate images, save, or quit. Coordinates are written
to a CSV (``image_path, point_index, z, y, x``). Before each save, the existing
CSV is copied into a ``backups/`` folder next to it with a timestamped name.

If the output CSV already exists, its annotations are loaded on startup so a
session can be resumed without losing earlier work.

Examples
--------
Annotate every CZI in a folder, writing to the default ``annotations.csv``::

    python annotate_centroids.py "data/*.czi"

Annotate a mix of explicit files and write to a chosen CSV path::

    python annotate_centroids.py img1.tif img2.czi --output runs/centroids.csv

Resume a previous session, jumping straight to the 12th image::

    python annotate_centroids.py "data/*.czi" \\
        --output runs/centroids.csv --start-index 12

The CSV path also acts as the load path: re-running with the same ``--output``
reloads prior points, and ``runs/backups/centroids_<timestamp>.csv`` snapshots
are kept on every save. You can also jump to any image with the "Go to image"
control and save mid-session with the "Save CSV" button.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import czifile
import napari
import numpy as np
import tifffile
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def parse_inputs(values: Sequence[str]) -> List[Path]:
    output: List[Path] = []
    for value in values:
        path = Path(value)
        if path.is_file():
            output.append(path)
        elif any(part in value for part in ["*", "?", "["]):
            glob_root = path.parent if path.parent.exists() else Path(".")
            output.extend(sorted(glob_root.glob(path.name)))
        else:
            raise FileNotFoundError(f"File not found: {value}")
    if not output:
        raise ValueError("No input files were found.")
    return output


def load_tiff(path: Path) -> np.ndarray:
    data = tifffile.imread(path)
    return normalize_image(data)


def load_czi(path: Path) -> np.ndarray:
    with czifile.CziFile(path) as czi:
        xarr = czi.asxarray()
    axes = "".join(xarr.dims)
    data = np.asarray(xarr)
    return normalize_image(data, axes=axes)


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return load_tiff(path)
    if suffix == ".czi":
        return load_czi(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_image(data: np.ndarray, axes: Optional[str] = None) -> np.ndarray:
    data = np.asarray(data)

    if axes is not None:
        axes = axes.upper()

        axis_map = {ax: idx for idx, ax in enumerate(axes)}
        # Keep only the first time point or mosaic tile if present.
        for extra in ["T", "S"]:
            if extra in axis_map:
                data = np.take(data, 0, axis=axis_map[extra])
                axes = axes.replace(extra, "")
                axis_map = {ax: idx for idx, ax in enumerate(axes)}

        target_axes = []
        for desired in ["Z", "Y", "X", "C"]:
            if desired in axis_map:
                target_axes.append(axis_map[desired])

        if target_axes:
            data = np.moveaxis(data, target_axes, list(range(len(target_axes))))
            if data.ndim == 3:
                return data
            if data.ndim == 4:
                # If no explicit C axis is present, squeeze a singleton axis to get a 3D image.
                if "C" not in axes:
                    singleton_axes = [idx for idx, size in enumerate(data.shape) if size == 1]
                    if len(singleton_axes) == 1:
                        return np.squeeze(data, axis=singleton_axes[0])
                # Ensure channel axis is last if we have an explicit C dim.
                if "C" in axes and (axes.endswith("C") or axes[axis_map["C"]] == "C"):
                    return data
                return np.moveaxis(data, -1, 3)
            return data

    return infer_and_normalize_image(data)


def infer_and_normalize_image(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        return data
    if data.ndim == 4:
        shape = data.shape
        singleton_axes = [axis for axis, size in enumerate(shape) if size == 1]
        if len(singleton_axes) == 1:
            squeezed = np.squeeze(data, axis=singleton_axes[0])
            if squeezed.ndim == 3:
                return squeezed
        channel_axes = [axis for axis, size in enumerate(shape) if size == 2]
        if len(channel_axes) == 1:
            return np.moveaxis(data, channel_axes[0], -1)
        if shape[-1] == 2:
            return data
        if shape[0] == 2:
            return np.moveaxis(data, 0, -1)
        return data
    if data.ndim == 5:
        shape = data.shape
        channel_axes = [axis for axis, size in enumerate(shape) if size == 2]
        if len(channel_axes) == 1:
            out = np.moveaxis(data, channel_axes[0], -1)
            return out.squeeze(axis=0) if out.shape[0] == 1 else out
        return data
    return data


class CentroidAnnotator:
    def __init__(
        self,
        file_paths: List[Path],
        output_path: Path,
        start_index: int = 0,
        backup_dir: Optional[Path] = None,
    ) -> None:
        self.file_paths = file_paths
        self.output_path = Path(output_path)
        self.backup_dir = Path(backup_dir) if backup_dir is not None else None
        self.annotations: Dict[str, np.ndarray] = {}

        # Load any previously saved annotations so that resuming a session does
        # not overwrite earlier work when the CSV is re-exported.
        self.load_existing_annotations()

        self.current_index = max(0, min(start_index, len(file_paths) - 1))

        self.viewer = napari.Viewer()
        self.image_layer = None
        self.points_layer = None
        self.status_label: Optional[QLabel] = None
        self.image_spinbox: Optional[QSpinBox] = None

        self.viewer.window.add_dock_widget(self.make_controls(), area="right")
        self.load_current_image()

    def make_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info_label = QLabel("Click to add centroids in the points layer.\nUse the buttons below to navigate and save.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Jump-to-image control so the session can resume on any image.
        jump_row = QWidget()
        jump_layout = QHBoxLayout()
        jump_layout.setContentsMargins(0, 0, 0, 0)
        jump_row.setLayout(jump_layout)
        jump_layout.addWidget(QLabel("Go to image:"))
        self.image_spinbox = QSpinBox()
        self.image_spinbox.setMinimum(1)
        self.image_spinbox.setMaximum(len(self.file_paths))
        self.image_spinbox.setValue(self.current_index + 1)
        go_button = QPushButton("Go")
        go_button.clicked.connect(lambda: self.jump_to_index(self.image_spinbox.value() - 1))
        jump_layout.addWidget(self.image_spinbox)
        jump_layout.addWidget(go_button)
        layout.addWidget(jump_row)

        next_button = QPushButton("Next image")
        prev_button = QPushButton("Previous image")
        save_progress_button = QPushButton("Save CSV")
        save_button = QPushButton("Save and quit")
        clear_button = QPushButton("Clear points")

        next_button.clicked.connect(lambda: (self.save_annotations(), self.change_index(1)))
        prev_button.clicked.connect(lambda: (self.save_annotations(), self.change_index(-1)))
        save_progress_button.clicked.connect(self.save_to_csv)
        save_button.clicked.connect(lambda: (self.save_to_csv(), self.viewer.close()))
        clear_button.clicked.connect(lambda: setattr(self.points_layer, "data", np.empty((0, 3))))

        layout.addWidget(prev_button)
        layout.addWidget(next_button)
        layout.addWidget(clear_button)
        layout.addWidget(save_progress_button)
        layout.addWidget(save_button)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        return widget

    def load_current_image(self) -> None:
        image_path = self.file_paths[self.current_index]
        image_data = load_image(image_path)

        channel_axis = -1 if image_data.ndim == 4 else None

        # Remove existing image layer(s) before re-adding, because channel_axis
        # can change between images and add_image returns a list when it is set.
        if self.image_layer is not None:
            layers_to_remove = self.image_layer if isinstance(self.image_layer, list) else [self.image_layer]
            for layer in layers_to_remove:
                self.viewer.layers.remove(layer)

        self.image_layer = self.viewer.add_image(
            image_data,
            name=image_path.name,
            channel_axis=channel_axis,
            contrast_limits=(np.nanmin(image_data), np.nanmax(image_data)),
        )

        # Remove and re-add the points layer so it is always on top.
        if self.points_layer is not None:
            self.viewer.layers.remove(self.points_layer)
        self.points_layer = self.viewer.add_points(
            self.annotations.get(str(image_path), np.empty((0, 3))),
            ndim=3,
            name="centroids",
            face_color="red",
            border_color="yellow",
            size=10,
            out_of_slice_display=True,
        )
        self.points_layer.mode = "add"
        self.points_layer.editable = True
        self.viewer.title = f"Centroid annotation ({self.current_index + 1}/{len(self.file_paths)})"

        # Keep the jump control in sync without re-triggering its handler.
        if self.image_spinbox is not None:
            self.image_spinbox.blockSignals(True)
            self.image_spinbox.setValue(self.current_index + 1)
            self.image_spinbox.blockSignals(False)
        self._update_status()

    def change_index(self, delta: int) -> None:
        next_index = self.current_index + delta
        if next_index < 0 or next_index >= len(self.file_paths):
            return
        self.current_index = next_index
        self.load_current_image()

    def jump_to_index(self, index: int) -> None:
        if index < 0 or index >= len(self.file_paths) or index == self.current_index:
            return
        self.save_annotations()
        self.current_index = index
        self.load_current_image()

    def save_annotations(self) -> None:
        image_path = self.file_paths[self.current_index]
        self.annotations[str(image_path)] = np.asarray(self.points_layer.data)

    def save_to_csv(self) -> None:
        """Persist current annotations to the CSV without closing the viewer."""
        self.save_annotations()
        self.export_csv(self.output_path)
        self._update_status(saved=True)

    def _update_status(self, saved: bool = False) -> None:
        if self.status_label is None:
            return
        image_path = self.file_paths[self.current_index]
        n_points = len(np.asarray(self.points_layer.data)) if self.points_layer else 0
        prefix = "Saved. " if saved else ""
        self.status_label.setText(
            f"{prefix}Image {self.current_index + 1}/{len(self.file_paths)}: {image_path.name} ({n_points} points)"
        )

    def load_existing_annotations(self) -> None:
        """Populate annotations from a previously saved CSV, if one exists."""
        if not self.output_path.exists():
            return
        loaded: Dict[str, List[List[float]]] = {}
        with self.output_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                loaded.setdefault(row["image_path"], []).append([float(row["z"]), float(row["y"]), float(row["x"])])
        for image_path, points in loaded.items():
            self.annotations[image_path] = np.asarray(points, dtype=float)

    def backup_csv(self) -> Optional[Path]:
        """Copy the current CSV into the backup folder with a timestamp."""
        if self.backup_dir is None or not self.output_path.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = self.backup_dir / f"{self.output_path.stem}_{timestamp}.csv"
        shutil.copy2(self.output_path, backup_path)
        return backup_path

    def export_csv(self, output_path: Path) -> None:
        # Snapshot the existing CSV before overwriting it.
        self.backup_csv()

        rows = []
        for image_path in self.file_paths:
            points = self.annotations.get(str(image_path), np.empty((0, 3)))
            for point_index, point in enumerate(points, start=1):
                rows.append(
                    {
                        "image_path": str(image_path),
                        "point_index": point_index,
                        "z": float(point[0]),
                        "y": float(point[1]),
                        "x": float(point[2]),
                    }
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "point_index", "z", "y", "x"])
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate centroid positions in TIFF and CZI image stacks.")
    parser.add_argument(
        "files",
        nargs="+",
        help="List of TIFF/CZI files or glob patterns to annotate.",
    )
    parser.add_argument(
        "--output",
        default="annotations.csv",
        help=(
            "CSV path for centroid coordinates. If it already exists, its "
            "annotations are loaded so the session can be resumed."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index of the image to resume annotation from.",
    )
    args = parser.parse_args()

    try:
        file_paths = parse_inputs(args.files)
    except Exception as exc:
        parser.error(str(exc))
        return 1

    output_path = Path(args.output)
    # Timestamped CSV backups always live in a 'backups' folder next to the output.
    backup_dir = output_path.parent / "backups"

    annotator = CentroidAnnotator(
        file_paths,
        output_path=output_path,
        start_index=args.start_index - 1,
        backup_dir=backup_dir,
    )
    napari.run()
    annotator.save_annotations()
    annotator.export_csv(output_path)
    print(f"Saved annotations to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
