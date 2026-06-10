"""Shelf dataset spec — train pseudo-label generation.

The standard Shelf benchmark uses frames [300, 600] for test; everything else (0-299
and 601-3199) is train.
"""

from pathlib import Path

from src.data.datasets.shelf_utils import get_cam_params as _get_cam_params

from preparation.utils.datasets.base import DatasetSpec


DATASET_PATH = Path("./data/shelf")
SEQUENCES_TRAIN = ["defalta"]                                     # single-sequence dataset
CAMERAS = [0, 1, 2, 3, 4]
TRAIN_FRAMES = list(range(0, 300)) + list(range(601, 3200))       # complement of test


def _image_path(seq: str, cam: int, frame: int) -> Path:
    return DATASET_PATH / f"Camera{cam}" / f"img_{frame:06d}.png"


def _enumerate_train_frames(seq: str):
    return iter(TRAIN_FRAMES)


def _cam_key(cam: int) -> int:
    return cam  # COMPOSE's ShelfDataset indexes detections by the raw int camera id


SPEC = DatasetSpec(
    name="shelf",
    dataset_path=DATASET_PATH,
    sequences=SEQUENCES_TRAIN,
    cameras=CAMERAS,
    n_joints=15,
    pixel_threshold=64.0**2,
    default_interval=1,                                          # use every train frame
    image_path=_image_path,
    enumerate_train_frames=_enumerate_train_frames,
    cam_key=_cam_key,
    get_cam_params=_get_cam_params,
    position_polygons={},
)
