"""Panoptic dataset — CMU Panoptic Studio."""

import time
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.paramUtil import PANOPTIC_CAM_CONFIGURATIONS

from .base import MultiViewPoseDataset
from .panoptic_utils import get_cam_params

log = logging.getLogger(__name__)


class PanopticDataset(MultiViewPoseDataset):
    ORIGINAL_IMAGE_SIZE = (1080, 1920)                       # (H, W)
    CAM_CONFIGURATIONS = PANOPTIC_CAM_CONFIGURATIONS
    get_cam_params = staticmethod(get_cam_params)

    @staticmethod
    def _read_image(
        img_path: str | Path,
        num_retries: int = 3,
        retry_sleep_s: float = 0.2,
    ) -> np.ndarray:
        """Robust read with retries + PIL fallback for shared-filesystem hiccups."""
        img_path = str(img_path)
        read_flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        for attempt in range(num_retries):
            img = cv2.imread(img_path, read_flags)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if attempt + 1 < num_retries:
                time.sleep(retry_sleep_s)
        try:
            with Image.open(img_path) as pil_img:
                return np.array(pil_img.convert("RGB"))
        except (FileNotFoundError, OSError):
            pass
        raise FileNotFoundError(f"Failed to load image after {num_retries} attempts: {img_path}")
