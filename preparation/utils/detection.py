"""Stage A — run RT-DETR + ViTPose++ on every frame of the train split.

The actual frame enumeration, image-path construction, and cam-key formatting all live
in the per-dataset `DatasetSpec` (see `preparation/utils/datasets/`).
"""

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    VitPoseForPoseEstimation,
)

from preparation.utils.datasets.base import DatasetSpec


VITPOSE_MODEL_NAME = "usyd-community/vitpose-plus-huge"
RTDETR_MODEL_NAME = "PekingU/rtdetr_v2_r101vd"


class PoseEstimator:
    """RT-DETR person detection + ViTPose++ pose estimation."""

    def __init__(self, rtdetr_model_name: str = RTDETR_MODEL_NAME, vitpose_model_name: str = VITPOSE_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.person_image_processor = RTDetrImageProcessor.from_pretrained(rtdetr_model_name, use_fast=True)
        self.person_model = RTDetrForObjectDetection.from_pretrained(rtdetr_model_name).to(self.device)  # type: ignore
        self.pose_image_processor = AutoImageProcessor.from_pretrained(vitpose_model_name, use_fast=True)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(vitpose_model_name).to(self.device)  # type: ignore

    @torch.no_grad()
    def detect(self, image: Image.Image) -> list[dict]:
        # RT-DETR person detection
        inputs = self.person_image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.person_model(**inputs)
        result = self.person_image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.5
        )[0]
        person_mask = result["labels"] == 0
        person_boxes = result["boxes"][person_mask]
        person_scores = result["scores"][person_mask]
        if person_boxes.shape[0] == 0:
            return []

        # xyxy → xywh
        bxywh = person_boxes.cpu().numpy().copy()
        bxywh[:, 2] -= bxywh[:, 0]
        bxywh[:, 3] -= bxywh[:, 1]

        # ViTPose++ keypoints
        inputs = self.pose_image_processor(images=image, boxes=[bxywh.tolist()], return_tensors="pt").to(self.device)
        outputs = self.pose_model(**inputs, dataset_index=torch.tensor([0], device=self.device))
        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs, boxes=[bxywh.tolist()])[0]

        scores = person_scores.cpu().numpy().tolist()
        for i, p in enumerate(pose_results):
            p["bbox_score"] = scores[i] if i < len(scores) else 0.0
            p["bbox"] = bxywh[i] if i < len(scores) else [0, 0, 0, 0]
        return pose_results


def _pack_view_detections(predictions: list[dict], n_keypoints: int) -> dict[str, np.ndarray]:
    if not predictions:
        return {
            "keypoints_xys": np.empty((0, n_keypoints, 3), dtype=np.float32),
            "bbox_xywhs": np.empty((0, 5), dtype=np.float32),
        }
    keypoints_xys = np.stack(
        [
            np.concatenate([p["keypoints"], np.asarray(p["scores"])[:, None]], axis=-1).astype(np.float32)
            for p in predictions
        ]
    )
    bbox_xywhs = np.stack(
        [np.array([*p["bbox"], p["bbox_score"]], dtype=np.float32) for p in predictions]
    )
    return {"keypoints_xys": keypoints_xys, "bbox_xywhs": bbox_xywhs}


def run_detection(spec: DatasetSpec, interval: int) -> dict[str, dict]:
    """Run detection over every (sequence, frame, camera) at `interval` stride.

    Returns: {sequence: {frame_num: {cam_key: {keypoints_xys, bbox_xywhs}}}}.
    """
    estimator = PoseEstimator()
    n_keypoints = len(estimator.pose_model.config.label2id)
    processed: dict[str, dict] = {}

    for seq in spec.sequences:
        train_frames = list(spec.enumerate_train_frames(seq))[::interval]
        processed[seq] = {}
        if not train_frames:
            print(f"Warning: no train frames found for {seq}, skipping")
            continue

        for frame_num in tqdm(train_frames, desc=f"Detecting {seq}"):
            joints_2d_data = {}
            for cam in spec.cameras:
                cam_str = spec.cam_key(cam)
                img_path = spec.image_path(seq, cam, frame_num)
                if not img_path.is_file():
                    joints_2d_data[cam_str] = _pack_view_detections([], n_keypoints)
                    continue
                try:
                    image = Image.open(img_path).convert("RGB")
                    joints_2d_data[cam_str] = _pack_view_detections(estimator.detect(image), n_keypoints)
                except Exception as e:
                    print(f"Error on {img_path}: {e}")
                    joints_2d_data[cam_str] = _pack_view_detections([], n_keypoints)

            processed[seq][frame_num] = {"2D": joints_2d_data}

    return processed
