"""Shelf dataset — 5-camera multi-person indoor scene."""

from src.utils.paramUtil import SHELF_CAM_CONFIGURATIONS

from .base import MultiViewPoseDataset
from .shelf_utils import get_cam_params


class ShelfDataset(MultiViewPoseDataset):
    ORIGINAL_IMAGE_SIZE = (776, 1032)                        # (H, W)
    CAM_CONFIGURATIONS = SHELF_CAM_CONFIGURATIONS
    get_cam_params = staticmethod(get_cam_params)

    # Hard-coded train/test frame split (standard Shelf benchmark protocol).
    def _compute_frame_interval(self, split: str, kwargs: dict) -> list:
        if split == "test":
            return list(range(300, 601))
        return list(range(0, 300)) + list(range(601, 3200))
