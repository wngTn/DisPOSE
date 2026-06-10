"""Campus dataset spec — train pseudo-label generation.

Standard Campus benchmark uses frames [350, 470] ∪ [650, 750] for test; train is the
complement: [0, 349] ∪ [471, 649] ∪ [751, 1999].
"""

from pathlib import Path

from src.data.datasets.campus_utils import get_cam_params as _get_cam_params

from preparation.utils.datasets.base import DatasetSpec


DATASET_PATH = Path("./data/campus")
SEQUENCES_TRAIN = ["defalta"]
CAMERAS = [0, 1, 2]
TRAIN_FRAMES = list(range(0, 350)) + list(range(471, 650)) + list(range(751, 2000))


def _image_path(seq: str, cam: int, frame: int) -> Path:
    return DATASET_PATH / f"Camera{cam}" / f"campus4-c{cam}-{frame:05d}.png"


def _enumerate_train_frames(seq: str):
    return iter(TRAIN_FRAMES)


def _cam_key(cam: int) -> int:
    return cam  # COMPOSE's CampusDataset indexes detections by the raw int camera id


SPEC = DatasetSpec(
    name="campus",
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
