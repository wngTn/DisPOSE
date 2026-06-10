"""Panoptic dataset spec — train pseudo-label generation."""

from pathlib import Path

from src.data.datasets.panoptic_utils import get_cam_params as _get_cam_params

from preparation.utils.datasets.base import DatasetSpec


DATASET_PATH = Path("./data/panoptic")
SEQUENCES_TRAIN = [
    "160422_ultimatum1",
    "160224_haggling1",
    "160226_haggling1",
    "161202_haggling1",
    "160906_ian1",
    "160906_ian2",
    "160906_ian3",
    "160906_band1",
    "160906_band2",
]
CAMERAS = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]


def _image_path(seq: str, cam: tuple[int, int], frame: int) -> Path:
    cam_str = f"{cam[0]:02d}_{cam[1]:02d}"
    return DATASET_PATH / seq / "hdImgs" / cam_str / f"{cam_str}_{frame:08d}.jpg"


def _enumerate_train_frames(seq: str):
    """Enumerate frame numbers from the panoptic 3D annotation files for this sequence."""
    ann_dir = DATASET_PATH / seq / "hdPose3d_stage1_coco19"
    if not ann_dir.exists():
        return
    for ann_file in sorted(ann_dir.iterdir()):
        if ann_file.is_file():
            try:
                yield int(ann_file.stem.split("_")[-1])
            except ValueError:
                continue


def _cam_key(cam: tuple[int, int]) -> str:
    return f"{cam[0]:02d}_{cam[1]:02d}"


# Per-camera "forbidden polygons" — dome edge regions where no valid persons appear.
POSITION_POLYGONS = {
    "00_03": [(0.00, 946.70), (80.70, 901.30), (172.83, 851.70), (335.78, 800.69),
              (527.10, 768.10), (403.80, 340.20), (137.40, 259.40), (0.00, 289.20)],
    "00_12": [(821.80, 537.10), (933.74, 513.04), (1115.12, 505.96), (1191.63, 521.54),
              (1224.22, 294.83), (1235.56, 144.63), (1085.30, 0.00), (857.20, 0.00),
              (777.90, 158.80), (749.54, 348.67)],
    "00_13": [(1484.95, 282.08), (1646.50, 374.20), (1802.35, 565.47), (1833.52, 673.16),
              (1920.00, 549.90), (1920.00, 0.00), (1712.70, 0.00), (1551.50, 0.00)],
    "00_06": [],
    "00_23": [],
}


SPEC = DatasetSpec(
    name="panoptic",
    dataset_path=DATASET_PATH,
    sequences=SEQUENCES_TRAIN,
    cameras=CAMERAS,
    n_joints=15,
    pixel_threshold=64.0**2,
    image_path=_image_path,
    enumerate_train_frames=_enumerate_train_frames,
    cam_key=_cam_key,
    get_cam_params=_get_cam_params,
    position_polygons=POSITION_POLYGONS,
)
