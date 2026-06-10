"""MM-OR dataset spec — train pseudo-label generation.

Reads source images / calibration from `./data/mm_or` (the shared mount). Writes
the pseudo-labels under `./data/preparation/mm_or/...` and visualizations under
`./preparation/output/mm_or/...`, like the other datasets.

Test split is fixed: {004_PKA, 011_TKA, 036_PKA, 038_TKA}. Train = every other
sequence available on disk.

Train frame numbers per sequence are determined dynamically from the intersection
of available images across all 5 cameras.
"""

from pathlib import Path

from src.data.datasets.mm_or_utils import get_cam_params as _get_cam_params

from preparation.utils.datasets.base import DatasetSpec
from preparation.utils.datasets.mm_or_polygons import FORBIDDEN_POLYGONS_BY_SEQUENCE


READ_PATH = Path("./data/mm_or")                                   # shared mount (read-only)
CAMERAS = [1, 2, 3, 4, 5]
TEST_SEQUENCES = {"004_PKA", "011_TKA", "036_PKA", "038_TKA"}


def _list_train_sequences() -> list[str]:
    """All sub-directories of READ_PATH except the test sequences.

    A sequence is included only if all required per-camera calibration files exist;
    a few sequences on the shared mount are missing them and would crash association.
    """
    if not READ_PATH.exists():
        return []
    out = []
    for d in sorted(READ_PATH.iterdir()):
        if not (d.is_dir() and d.name not in TEST_SEQUENCES):
            continue
        if not all((d / f"camera{c:02d}.json").exists() for c in CAMERAS):
            continue
        out.append(d.name)
    return out


SEQUENCES_TRAIN = _list_train_sequences()


def _image_path(seq: str, cam: int, frame: int) -> Path:
    return READ_PATH / seq / "colorimage" / f"camera{cam:02d}_colorimage-{frame:06d}.jpg"


def _enumerate_train_frames(seq: str):
    """Frames available across all 5 cameras for this sequence."""
    per_cam = []
    for cam in CAMERAS:
        cam_dir = READ_PATH / seq / "colorimage"
        rgb_imgs = sorted(cam_dir.glob(f"camera{cam:02d}_colorimage-*.jpg"))
        per_cam.append({int(p.stem.split("-")[-1]) for p in rgb_imgs})
    if not per_cam:
        return iter([])
    return iter(sorted(set.intersection(*per_cam)))


def _cam_key(cam: int) -> int:
    return cam  # COMPOSE's MMORDataset indexes detections by the raw int camera id


SPEC = DatasetSpec(
    name="mm_or",
    dataset_path=READ_PATH,
    sequences=SEQUENCES_TRAIN,
    cameras=CAMERAS,
    n_joints=15,
    pixel_threshold=64.0**2,
    image_path=_image_path,
    enumerate_train_frames=_enumerate_train_frames,
    cam_key=_cam_key,
    get_cam_params=_get_cam_params,
    position_polygons={},
    # The OR patient lies on an elevated table; drop any person whose triangulated
    # ankle is more than 0.25 m above the floor.
    patient_foot_z_threshold_mm=250.0,
    # Per-sequence forbidden regions (static equipment/monitors) culled before association.
    forbidden_polygons_by_sequence=FORBIDDEN_POLYGONS_BY_SEQUENCE,
)
