"""Base class for multi-view multi-person 3D pose datasets.

The four concrete datasets (`PanopticDataset`, `ShelfDataset`, `CampusDataset`,
`MMORDataset`) only differ in:

* the dataset-level constants `ORIGINAL_IMAGE_SIZE` and `CAM_CONFIGURATIONS`,
* the choice of `get_cam_params` (which lives in the dataset's `_utils.py`),
* whether `_read_image` retries on transient I/O failures (Panoptic / MM-OR do; the
  smaller Shelf / Campus datasets do not),
* and a split-specific `_compute_frame_interval` for Shelf and Campus.

All other logic — augmentation sampling, valid-region checks, keypoint affine
transforms, batch packing — is identical across the four.
"""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from src.data.utils import get_scale
from src.utils.linear_algebra import get_affine_transform
from src.utils.paramUtil import KEYPOINT_INFO

log = logging.getLogger(__name__)


class MultiViewPoseDataset(data.Dataset):
    """Shared base for multi-view multi-person 3D pose datasets.

    Subclasses must set:
        ORIGINAL_IMAGE_SIZE: tuple[int, int]                — (H, W) of the raw frames
        CAM_CONFIGURATIONS: dict[str, list]                 — camera_setup_name → list of cameras
        get_cam_params: Callable[[sequences, data_root, cam_list], dict[str, np.ndarray]]
                                                            — typically imported from `<dataset>_utils`

    Subclasses may override:
        _read_image: per-frame image loader (defaults to simple cv2 read)
        _compute_frame_interval(split, kwargs): split-specific frame ranges
    """

    ORIGINAL_IMAGE_SIZE: tuple[int, int]
    CAM_CONFIGURATIONS: dict
    get_cam_params: Callable

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    @staticmethod
    def _read_image(img_path: str | Path) -> np.ndarray:
        """Read an RGB image. Default: simple cv2 read, raise on failure."""
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @classmethod
    def _load_image(
        cls,
        img_paths: list,
        affine_transforms: np.ndarray,
        target_img_size: np.ndarray,
    ) -> np.ndarray:
        """Load multi-view images and apply per-view affine transforms.

        Args:
            img_paths: list of V image paths
            affine_transforms: (V, A, 2, 3)
            target_img_size: (W, H)

        Returns:
            (V, A, C, H, W)
        """
        V = len(img_paths)
        tgt_w, tgt_h = int(target_img_size[0]), int(target_img_size[1])

        source = []
        for v in range(V):
            try:
                img = cls._read_image(img_paths[v])
            except FileNotFoundError:
                # Shared storage can fail intermittently; substitute blanks rather than
                # crashing the training job. Subclasses that want strict failure can
                # override `_read_image` to not raise FileNotFoundError.
                log.warning("Using blank fallback image for unreadable path: %s", img_paths[v])
                blank = np.zeros((len(affine_transforms[v]), 3, tgt_h, tgt_w), dtype=np.uint8)
                source.append(blank)
                continue

            augmented_imgs = [
                cv2.warpAffine(img, aff_trans, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
                for aff_trans in affine_transforms[v]
            ]
            # (A, H, W, C) → (A, C, H, W)
            source.append(np.array(augmented_imgs).transpose(0, 3, 1, 2))

        return np.array(source)

    # ------------------------------------------------------------------
    # Subclass extension hook
    # ------------------------------------------------------------------
    def _compute_frame_interval(self, split: str, kwargs: dict) -> list | None:
        """Default: forward to the caller's `frame_interval` kwarg (Panoptic / MM-OR style)."""
        return kwargs.get("frame_interval")

    # ------------------------------------------------------------------
    # Init / length / getitem
    # ------------------------------------------------------------------
    def __init__(
        self,
        mean: tuple,
        std: tuple,
        split: Literal["train", "test"],
        loader,
        transforms,
        collate_fn,
        data_dir: str,
        augmentations: dict,
        num_temporal: int,
        interval: int,
        camera_setup: str,
        skeleton_type: str,
        image_size: tuple,
        heatmap_size: tuple,
        sequences: list[str],
        include_identity_slot: bool = True,
        **kwargs,
    ):
        if num_temporal % 2 == 0:
            raise ValueError("`num_temporal` must be odd so the center frame is the target.")

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.split = split
        self.collate_fn = collate_fn
        self.data_root = Path(data_dir)
        self.augmentations = augmentations
        self.num_temporal = num_temporal
        self.interval = interval
        self.njoints = len(KEYPOINT_INFO[skeleton_type])
        self.original_image_size = self.ORIGINAL_IMAGE_SIZE
        self.image_size = np.array(image_size, dtype=np.float32)
        self.heatmap_size = np.array(heatmap_size, dtype=np.float32)
        self.sequences = sequences
        self.transform = transforms
        self.skeleton_type = skeleton_type
        self.frame_interval = self._compute_frame_interval(split, kwargs)

        # Augmentation slots. With include_identity_slot=True (default, used by the
        # pose task's cross-affine loss), slot 0 is always identity and len(rotation_factors)
        # more slots get random rotation/scale/flip. With include_identity_slot=False (used
        # by backbone fine-tuning), every slot is random.
        self.include_identity_slot = bool(include_identity_slot)
        n_random_augs = len(self.augmentations.get("rotation_factors", []))
        self.num_augs = n_random_augs + (1 if self.include_identity_slot else 0)
        if self.num_augs == 0:
            self.num_augs = 1
            self.include_identity_slot = True

        self.cam_list = list(self.CAM_CONFIGURATIONS[camera_setup])

        self.max_num_people = 10
        self.root_idx = 2
        self.max_aug_tries = 100

        self.cam_params_per_sequence = type(self).get_cam_params(
            self.sequences,
            self.data_root,
            self.cam_list,
        )

        self.sequences_data, self.frame_map = loader(
            cam_params_per_sequence=self.cam_params_per_sequence,
            cam_list=self.cam_list,
            data_root=self.data_root,
            sequences=self.sequences,
            interval=self.interval,
            frame_interval=self.frame_interval,
            root_idx=self.root_idx,
        )()

    def __len__(self):
        return len(self.frame_map)

    def __getitem__(self, idx):
        sequence_name, target_frame_idx = self.frame_map[idx]

        # Centered temporal window
        half = (self.num_temporal - 1) // 2
        sequence_info = self.sequences_data[sequence_name]
        num_frames_in_seq = len(sequence_info)
        start = target_frame_idx - half
        end = target_frame_idx + half
        frame_indices = np.clip(np.arange(start, end + 1), 0, num_frames_in_seq - 1).tolist()

        sequence_data = []
        for frame_idx in frame_indices:
            frame_data = sequence_info[frame_idx].copy()

            aug_results = self.compute_augmentations(
                frame_data["gt_keypoints_xys"],
                H=self.original_image_size[0],
                W=self.original_image_size[1],
            )

            source = self._load_image(
                frame_data["img_paths"],
                aug_results["affine_transforms"],
                self.image_size,
            )

            # Per-augmentation color/intensity transforms.
            augmented_source = np.zeros_like(source)
            for v in range(len(self.cam_list)):
                for a in range(self.num_augs):
                    img_array = source[v, a].transpose(1, 2, 0).astype(np.uint8)
                    aug_img = self.transform(Image.fromarray(img_array))
                    augmented_source[v, a] = np.array(aug_img).transpose(2, 0, 1)
            source = augmented_source

            source = source / 255.0
            source = (source - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)

            frame_data["source"] = source
            frame_data.update(aug_results)

            # Pad to max_num_people
            num_pad = self.max_num_people - frame_data["gt_keypoints_xyzs"].shape[0]
            frame_data["gt_keypoints_xyzs"] = np.pad(
                frame_data["gt_keypoints_xyzs"], ((0, num_pad), (0, 0), (0, 0)),
            )
            frame_data["global_ids"] = np.pad(
                frame_data["global_ids"], (0, num_pad), constant_values=-1,
            )
            num_pad = self.max_num_people - frame_data["person_ids"].shape[1]
            frame_data["person_ids"] = np.pad(
                frame_data["person_ids"], ((0, 0), (0, num_pad)), constant_values=-1,
            )

            sequence_data.append(frame_data)

        return_value = defaultdict(list)
        for d in sequence_data:
            for k, v in d.items():
                return_value[k].append(v)

        return self.pack_batch(return_value, sequence_name)

    # ------------------------------------------------------------------
    # Augmentation helpers (identical across the 4 datasets)
    # ------------------------------------------------------------------
    def _sample_augmentation_params(self):
        rot_factor = random.choice(self.augmentations.get("rotation_factors", [0]))
        flip_factor = random.choice(self.augmentations.get("flip_factors", [False]))
        scale_factor = random.choice(self.augmentations.get("scale_factors", [0]))

        rotation = (
            np.clip(np.random.uniform(-1, 1) * rot_factor, -rot_factor, rot_factor)
            if random.random() <= 0.5 and rot_factor != 0
            else 0.0
        )

        scale_val = 0.0
        if scale_factor != 0 and random.random() <= 0.5:
            scale_val = (
                np.random.uniform(0.1, scale_factor)
                if random.random() <= 0.5
                else -np.random.uniform(0.1, scale_factor) / 2.0
            )

        hflip = random.random() <= 0.5 and flip_factor
        return rotation, scale_val, hflip

    def _has_visible_person(self, keypoints: np.ndarray) -> bool:
        if keypoints.shape[0] == 0:
            return True
        return (keypoints[..., 2] > 0).any(axis=1).any()

    def _point_in_rotated_rect(self, points: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Mask of points inside a quadrilateral defined by 4 ordered corners."""
        original_shape = points.shape[:-1]
        points_2d = points.reshape(-1, 2)
        inside = np.ones(points_2d.shape[0], dtype=bool)
        for i in range(4):
            p1, p2 = corners[i], corners[(i + 1) % 4]
            edge = p2 - p1
            to_point = points_2d - p1
            cross = edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0]
            inside &= cross >= 0
        return inside.reshape(original_shape)

    def _get_valid_image_region(
        self, affine_transform: np.ndarray, orig_w: int, orig_h: int
    ) -> np.ndarray:
        """Corners of the valid (non-black-padding) region after the affine transform."""
        corners_orig = np.array(
            [[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]], dtype=np.float32
        )
        corners_homog = np.concatenate(
            [corners_orig, np.ones((4, 1), dtype=np.float32)], axis=1
        )
        return corners_homog @ affine_transform.T

    def _apply_augmentation_to_keypoints(
        self,
        keypoints: np.ndarray,
        affine_transform: np.ndarray,
        hflip: bool,
        img_w: int,
        img_h: int,
        orig_w: int,
        orig_h: int,
        pad_left: int = 25,
        pad_right: int = 25,
    ) -> np.ndarray:
        N, J, _ = keypoints.shape
        result = keypoints.copy()
        if N == 0:
            return result

        original_score = keypoints[..., 2]
        pts_xy = keypoints[..., :2]
        pts_homog = np.concatenate([pts_xy, np.ones((N, J, 1), dtype=pts_xy.dtype)], axis=-1)
        transformed_pts = np.einsum("njd,bid -> nji", pts_homog, affine_transform)

        x_coords, y_coords = transformed_pts[..., 0], transformed_pts[..., 1]
        within_output_bounds = (
            (x_coords >= pad_left) & (y_coords >= 0)
            & (x_coords < (img_w - pad_right)) & (y_coords < img_h)
        )
        valid_region = self._get_valid_image_region(affine_transform[0], orig_w, orig_h)
        within_valid_region = self._point_in_rotated_rect(transformed_pts, valid_region)

        new_visibility = original_score * within_output_bounds * within_valid_region
        result[..., :2] = transformed_pts
        result[..., 2] = new_visibility.astype(result.dtype)

        if hflip:
            raise NotImplementedError("Horizontal flip not implemented yet")
        return result

    def _find_valid_augmentation(self, keypoints_orig: np.ndarray, H: int, W: int):
        center = np.array([W / 2, H / 2], dtype=np.float32)
        base_scale = get_scale((W, H), self.image_size)
        img_w, img_h = int(self.image_size[0]), int(self.image_size[1])

        orig_has_visible = self._has_visible_person(keypoints_orig)
        for _ in range(self.max_aug_tries):
            rotation, scale_val, hflip = self._sample_augmentation_params()
            scale = base_scale + base_scale * scale_val
            affine_transform = get_affine_transform(
                torch.from_numpy(center),
                torch.from_numpy(scale),
                torch.tensor(rotation, dtype=torch.float32),
                torch.from_numpy(self.image_size),
            ).numpy()
            aug_keypoints = self._apply_augmentation_to_keypoints(
                keypoints_orig, affine_transform, hflip, img_w, img_h, W, H,
            )
            aug_has_visible = self._has_visible_person(aug_keypoints)
            if not orig_has_visible or aug_has_visible:
                return rotation, scale_val, hflip, scale, affine_transform, aug_keypoints
        return None

    def compute_augmentations(self, joints_xys: np.ndarray, H: int, W: int) -> dict:
        V, N_orig, J, _ = joints_xys.shape
        assert J == self.njoints, f"Expected {self.njoints} joints, got {J}"

        nposes = min(N_orig, self.max_num_people)
        A = self.num_augs

        keypoints_transformed = np.zeros((V, A, self.max_num_people, J, 3), dtype=np.float32)
        keypoints_original_space = np.zeros((V, A, self.max_num_people, J, 3), dtype=np.float32)
        centers = np.zeros((V, A, 2), dtype=np.float32)
        scales = np.zeros((V, A, 2), dtype=np.float32)
        rotations = np.zeros((V, A, 1), dtype=np.float32)
        hflips = np.zeros((V, A, 1), dtype=np.bool_)
        affine_transforms = np.zeros((V, A, 2, 3), dtype=np.float32)

        center = np.array([W / 2, H / 2], dtype=np.float32)
        base_scale = get_scale((W, H), self.image_size)
        img_w, img_h = int(self.image_size[0]), int(self.image_size[1])

        for v in range(V):
            view_keypoints = joints_xys[v, :nposes].copy()

            identity_affine = get_affine_transform(
                torch.from_numpy(center),
                torch.from_numpy(base_scale),
                torch.tensor(0.0),
                torch.from_numpy(self.image_size),
            ).numpy()
            identity_keypoints_transformed = self._apply_augmentation_to_keypoints(
                view_keypoints, identity_affine, False, img_w, img_h, W, H,
            )

            if self.include_identity_slot:
                keypoints_transformed[v, 0, :nposes] = identity_keypoints_transformed
                keypoints_original_space[v, 0, :nposes] = view_keypoints
                centers[v, 0] = center
                scales[v, 0] = base_scale
                rotations[v, 0, 0] = 0.0
                hflips[v, 0, 0] = False
                affine_transforms[v, 0] = identity_affine
                first_random_slot = 1
            else:
                first_random_slot = 0

            for a in range(first_random_slot, A):
                result = self._find_valid_augmentation(view_keypoints, H, W)
                if result is None:
                    keypoints_transformed[v, a, :nposes] = identity_keypoints_transformed
                    keypoints_original_space[v, a, :nposes] = view_keypoints
                    centers[v, a] = center
                    scales[v, a] = base_scale
                    rotations[v, a, 0] = 0.0
                    hflips[v, a, 0] = False
                    affine_transforms[v, a] = identity_affine
                else:
                    rotation, scale_val, hflip, scale, affine_transform, aug_kp = result
                    keypoints_transformed[v, a, :nposes] = aug_kp
                    # Original-space copy: same xy as the input, visibility from the aug
                    orig_space_kpts = view_keypoints.copy()
                    orig_space_kpts[..., 2] = aug_kp[..., 2]
                    keypoints_original_space[v, a, :nposes] = orig_space_kpts
                    centers[v, a] = center
                    scales[v, a] = scale
                    rotations[v, a, 0] = rotation
                    hflips[v, a, 0] = hflip
                    affine_transforms[v, a] = affine_transform

        return {
            "gt_keypoints_xys": keypoints_original_space,
            "gt_keypoints_xys_transformed": keypoints_transformed,
            "center": centers,
            "scale": scales,
            "rotation": rotations,
            "hflip": hflips,
            "affine_transforms": affine_transforms,
        }

    def pack_batch(self, return_value: dict, sequence_name: str) -> dict:
        """Repack to convention: (A, T, V, ...) per-view data, (A, T, ...) otherwise."""
        keys_with_VA = {
            "source", "center", "scale", "rotation", "hflip", "affine_transforms",
            "gt_keypoints_xys", "gt_keypoints_xys_transformed",
        }
        keys_T_only = {
            "gt_keypoints_xyzs", "person_ids", "global_ids", "frame_num", "cam_params_vec",
        }
        keys_TV_noA = {"img_paths"}

        final_batch = {"sequence": sequence_name}
        for k, v in return_value.items():
            if k == "sequence":
                continue
            if k == "img_paths":
                arr = np.array(v, dtype=object)
            else:
                arr = np.array(v)
                if arr.dtype != object:
                    arr = arr.astype(np.int64 if k == "frame_num" else np.float32)

            if k in keys_with_VA:
                if arr.ndim < 3:
                    raise ValueError(f"Key '{k}' expected ≥3 dims, got {arr.shape}")
                arr = np.transpose(arr, (2, 0, 1, *range(3, arr.ndim)))
            elif k in keys_TV_noA or k in keys_T_only:
                arr = np.expand_dims(arr, axis=0).repeat(self.num_augs, 0)
            else:
                arr = np.expand_dims(arr, axis=0).repeat(self.num_augs, 0)

            final_batch[k] = arr
        return final_batch
