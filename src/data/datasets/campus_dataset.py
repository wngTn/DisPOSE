"""Campus dataset — 3-camera outdoor courtyard scene."""

from src.utils.paramUtil import CAMPUS_CAM_CONFIGURATIONS

from .base import MultiViewPoseDataset
from .campus_utils import get_cam_params


class CampusDataset(MultiViewPoseDataset):
    ORIGINAL_IMAGE_SIZE = (288, 360)                         # (H, W)
    CAM_CONFIGURATIONS = CAMPUS_CAM_CONFIGURATIONS
    get_cam_params = staticmethod(get_cam_params)

    # Hard-coded train/test frame split (standard Campus benchmark protocol).
    def _compute_frame_interval(self, split: str, kwargs: dict) -> list:
        if split == "test":
            return list(range(350, 471)) + list(range(650, 751))
        return list(range(0, 350)) + list(range(471, 650)) + list(range(751, 2000))
