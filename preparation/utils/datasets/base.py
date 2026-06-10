"""DatasetSpec — the per-dataset configuration shape used by the prepare pipeline.

Each concrete dataset module (`panoptic.py`, `shelf.py`, `campus.py`, `mm_or.py`)
constructs a `DatasetSpec` and exports it as `SPEC`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


@dataclass
class DatasetSpec:
    name: str
    dataset_path: Path                                             # where source images / calibration live (may be a shared mount)
    sequences: list                                                # train sequences (per-dataset typed)
    cameras: list                                                  # tuples (panoptic) or ints (others)
    n_joints: int                                                  # output skeleton joint count
    pixel_threshold: float                                         # worst-view-dropout error cap (px²)

    # Helpers — closed-over per-dataset constants.
    image_path: Callable[[str, object, int], Path]                 # (sequence, camera, frame) → image path
    enumerate_train_frames: Callable[[str], Iterable[int]]         # sequence → iterable of train frame numbers
    cam_key: Callable[[object], str | int]                         # camera → cam_id (str for panoptic, int for others) used as pkl key
    get_cam_params: Callable[[list, Path, list], dict[str, np.ndarray]]  # (sequences, root, cameras) → {seq: (V, 51)}

    # ---- defaulted fields (must come after non-defaulted fields per @dataclass rules) ----
    default_interval: int = 3                                      # frame stride used when --interval is not provided

    # Optional position-filter polygons (only Panoptic has hand-marked dome dead zones).
    position_polygons: dict[str, list] = field(default_factory=dict)

    # Optional override for where pseudo-label and viz output goes.
    # Default = data/preparation/<name>/ and preparation/output/<name>/ (set by RunPaths).
    # MM-OR overrides this to a local writable dir because data/mm_or is shared read-only.
    output_labels_base: Path | None = None
    output_viz_base: Path | None = None

    # Optional patient filter (MM-OR): drop any person whose triangulated ankle rises
    # above this height (mm above the z=0 ground). The patient lies on an elevated
    # operating table, so their feet are well above the standing staff. None disables it.
    patient_foot_z_threshold_mm: float | None = None

    # Optional per-sequence forbidden polygons (MM-OR): {sequence: {camera_int: [[(x, y), ...], ...]}}.
    # Drops 2D detections that fall (mostly) inside hand-marked equipment/monitor regions.
    # When set, the per-camera `position_polygons` above is ignored for this dataset.
    forbidden_polygons_by_sequence: dict | None = None
