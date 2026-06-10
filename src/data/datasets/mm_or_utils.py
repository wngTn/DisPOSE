"""
Utility functions for the MM-OR dataset
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from src.utils.camera import world_3d_to_img_2d
from src.utils.common import NoOp, get_rank

ANNOTATION_FRAMES = {
    "004_PKA": [0, 300],
    "011_TKA": [0, 300],
    "036_PKA": [0, 150],
}


class GTDataLoader:
    """Loader for ground truth data from Panoptic dataset."""

    def __init__(
        self,
        cam_params_per_sequence: dict[str, np.ndarray],
        cam_list: list[tuple],
        data_root: Path,
        sequences: list[str],
        interval: int = 1,
        frame_interval: tuple | None = None,
        root_idx: int = 0,
        load_unlabeled_frames: bool = False,
    ):
        self.cam_params_per_sequence = cam_params_per_sequence
        self.cam_list = cam_list
        self.data_root = data_root
        self.sequences = sequences
        self.interval = interval
        self.frame_interval = frame_interval
        self.root_idx = root_idx
        self.load_unlabeled_frames = load_unlabeled_frames

    def _process_frame(
        self,
        sequence: str,
        frame_num: int,
        cam_params: np.ndarray,
        global_id_counter: int,
        annotation_file: Path | None = None,
    ) -> tuple[dict | None, int]:
        """Process a single frame. Ignores frame if ANY camera image is missing."""

        # 1. Validate Image Existence FIRST
        # If any camera view is missing for this frame, we skip the entire frame.
        img_paths = []
        for cam_view in self.cam_list:
            img_path = self.data_root / sequence / "colorimage" / f"camera{cam_view:02d}_colorimage-{frame_num:06d}.jpg"
            if not img_path.exists():
                # Missing image for one of the views -> Ignore this frame completely
                return None, global_id_counter
            img_paths.append(img_path)

        # 2. Process Annotations (if they exist)
        gt_keypoints_xyzs, person_ids, global_ids = [], [], []

        if annotation_file and annotation_file.exists():
            with annotation_file.open("r", encoding="utf-8") as f:
                annotation_3d = json.load(f)

            for annotation in annotation_3d:
                person_id = annotation["id"]
                joints_3d = np.array(annotation["keypoints_xyzs"], dtype=np.float32).reshape(15, 4)

                if joints_3d[self.root_idx, -1] <= 0.1:
                    continue

                gt_keypoints_xyzs.append(joints_3d)
                person_ids.append(person_id)
                # Assign global ID
                global_ids.append(global_id_counter)
                global_id_counter += 1

        # 3. Handle Empty GT (Unlabeled config or empty JSON)
        if not gt_keypoints_xyzs:
            if not self.load_unlabeled_frames:
                # If we are strictly loading labeled data and found none, skip.
                return None, global_id_counter

            # Prepare empty arrays for unlabeled frame
            gt_keypoints_xyzs = np.zeros((0, 15, 4), dtype=np.float32)
            person_ids = np.zeros((0,), dtype=int)
            global_ids = np.zeros((0,), dtype=int)
        else:
            gt_keypoints_xyzs = np.array(gt_keypoints_xyzs)
            person_ids = np.array(person_ids)
            global_ids = np.array(global_ids)

        # 4. Expand and Project
        # Expand IDs: (N,) -> (NumCams, N)
        person_ids = person_ids[None].repeat(cam_params.shape[0], 0)

        # Project 3D -> 2D
        if len(gt_keypoints_xyzs) > 0:
            gt_keypoints_xy, gt_keypoints_valid = world_3d_to_img_2d(
                torch.from_numpy(gt_keypoints_xyzs[..., :3]),
                torch.from_numpy(cam_params),
            )
            gt_keypoints_xy = gt_keypoints_xy.numpy()
            gt_keypoints_valid = gt_keypoints_valid.numpy()
        else:
            gt_keypoints_xy = np.zeros((cam_params.shape[0], 0, 15, 2), dtype=np.float32)
            gt_keypoints_valid = np.zeros((cam_params.shape[0], 0, 15, 1), dtype=np.float32)

        # Prepare 2.5D keypoints
        gt_keypoints_s = gt_keypoints_xyzs[None, ..., -1:]  # (1, N, 15, 1)
        gt_keypoints_s = gt_keypoints_s.repeat(cam_params.shape[0], 0)  # (NumCams, N, 15, 1)

        # Concatenate (X, Y, S*Valid)
        gt_keypoints_xys = np.concatenate([gt_keypoints_xy, gt_keypoints_s * gt_keypoints_valid], axis=-1)

        frame_data = {
            "sequence": sequence,
            "frame_num": frame_num,
            "gt_keypoints_xyzs": gt_keypoints_xyzs,
            "gt_keypoints_xys": gt_keypoints_xys,
            "person_ids": person_ids,
            "global_ids": global_ids,
            "img_paths": img_paths,  # Already verified and collected at step 1
            "cam_params_vec": cam_params,
        }

        return frame_data, global_id_counter

    def __call__(self) -> tuple[dict, list[tuple]]:
        """Load ground truth data from annotation files or image scans."""
        sequences_data = {}
        frame_map = []
        global_id_counter = 0

        for sequence in self.sequences:
            cam_params = self.cam_params_per_sequence[sequence]
            current_sequence_frames = []

            frames_to_load = []  # List of (frame_num, annotation_path)

            # --- Strategy: Gather Frame Numbers ---
            if self.load_unlabeled_frames:
                # Strategy A: Scan Images (using 1st camera)
                ref_cam = self.cam_list[0] if self.cam_list else 0
                image_dir = self.data_root / sequence / "colorimage"

                if image_dir.exists():
                    # Pattern: camera00_colorimage-00000123.jpg
                    for img_file in sorted(image_dir.glob(f"camera{ref_cam:02d}_colorimage-*.jpg")):
                        try:
                            frame_num = int(img_file.stem.split("-")[-1])
                            # Construct hypothetical pose path (it might not exist)
                            annotation_path = self.data_root / sequence / "poses" / f"body3DScene_{frame_num:08d}.json"
                            frames_to_load.append((frame_num, annotation_path))
                        except ValueError:
                            continue
            else:
                # Strategy B: Scan Poses (Standard)
                annotation_dir = self.data_root / sequence / "poses"
                if annotation_dir.exists():
                    annotation_list = sorted(annotation_dir.iterdir())

                    if sequence in ANNOTATION_FRAMES:
                        frame_range = ANNOTATION_FRAMES[sequence]
                        annotation_list = annotation_list[frame_range[0] : frame_range[1]]

                    for ann_file in annotation_list:
                        try:
                            # Pattern: body3DScene_00000123.json
                            frame_num = int(ann_file.stem.split("_")[-1])
                            frames_to_load.append((frame_num, ann_file))
                        except ValueError:
                            continue

            if not frames_to_load:
                continue

            # --- Processing Loop ---
            rank = get_rank()
            pbar = tqdm(total=len(frames_to_load), desc=f"Loading {sequence}...") if rank == 0 else NoOp()

            for frame_num, annotation_path in frames_to_load:
                if self.frame_interval is not None:
                    if frame_num not in range(self.frame_interval[0], self.frame_interval[1] + 1):
                        pbar.update(1)
                        continue

                frame_data, global_id_counter = self._process_frame(
                    sequence=sequence,
                    frame_num=frame_num,
                    cam_params=cam_params,
                    global_id_counter=global_id_counter,
                    annotation_file=annotation_path,
                )

                if frame_data is not None:
                    current_sequence_frames.append(frame_data)

                pbar.update(1)

            pbar.close()

            if not current_sequence_frames:
                continue

            sequences_data[sequence] = current_sequence_frames

            for frame_idx_in_seq in range(len(current_sequence_frames)):
                frame_map.append((sequence, frame_idx_in_seq))

        return sequences_data, frame_map


class PseudoGTDataLoader:
    """Loader for pseudo ground truth data with filtering capabilities."""

    def __init__(
        self,
        label_file: str,
        min_visible_joints: int,
        cam_params_per_sequence: dict[str, np.ndarray],
        cam_list: list[int],
        data_root: Path,
        sequences: list[str],
        interval: int,
        filter_unmatched: bool = True,
        frame_interval: tuple | None = None,
        root_idx: int = 2,
        max_people: int = 10,
    ):
        self.label_file = Path(label_file)
        self.cam_params_per_sequence = cam_params_per_sequence
        self.cam_list = cam_list
        self.data_root = data_root
        self.sequences = sequences
        self.interval = interval
        self.filter_unmatched = filter_unmatched
        self.frame_interval = frame_interval
        self.root_idx = root_idx
        self.min_visible_joints = min_visible_joints
        self.max_people = max_people

        self.num_joints = 15
        self.num_cams = len(cam_list)

    def _is_valid_skeleton(self, keypoints_xyz: np.ndarray) -> bool:
        """Check if a 3D skeleton has enough valid joints and is not the patient."""
        # 1. Existing check: Minimum visible joints
        valid_mask = ~np.all(keypoints_xyz == 0, axis=1)
        num_valid = np.sum(valid_mask)

        if num_valid < self.min_visible_joints:
            return False

        # Filter out patient skeletons based on ankle height
        # Indices: 8 (Left Ankle), 14 (Right Ankle)
        # We assume (0,0,0) indicates a missing joint. Since 0 < 350, missing joints will naturally pass this check (we only filter confirmed high ankles).
        left_ankle_z = keypoints_xyz[8, 2]
        right_ankle_z = keypoints_xyz[14, 2]
        if left_ankle_z > 250 or right_ankle_z > 250:
            return False

        return True

    def _process_frame(
        self,
        sequence: str,
        frame_num: int,
        frame_2d_data: dict,
        frame_3d_data: dict | None,
        cam_params: np.ndarray,
        global_id_counter: int,
    ) -> tuple[dict | None, int]:
        """Process a single frame and return filtered data."""

        # Parse 3D data
        if frame_3d_data and frame_3d_data["person_ids"].size:
            person_ids_3d = frame_3d_data["person_ids"]
            triangulated_xyzs = frame_3d_data["triangulated_keypoints_xyzs"]
            order = np.argsort(person_ids_3d)
            pid_to_3d = {pid: triangulated_xyzs[i] for i, pid in enumerate(person_ids_3d[order])}
        else:
            pid_to_3d = {}

        # Collect 2D detections per person per view
        pid_to_cams: dict[int, dict[int, np.ndarray]] = {}
        for cam_idx, cam_key in enumerate(self.cam_list):
            if str(cam_key) not in frame_2d_data:
                continue
            data_src = frame_2d_data[str(cam_key)]
            for pid, kps_xy in zip(data_src["person_ids"], data_src["keypoints_xys"]):
                if pid not in pid_to_cams:
                    pid_to_cams[pid] = {}
                pid_to_cams[pid][cam_idx] = kps_xy

        if not pid_to_cams:
            return None, global_id_counter

        # Categorize detections
        valid_matched: list[tuple[int, np.ndarray, dict[int, np.ndarray]]] = []
        invalid_matched: list[tuple[int, dict[int, np.ndarray]]] = []
        unmatched: list[tuple[float, int, np.ndarray]] = []  # (confidence, cam_idx, kps_xy)

        for pid, cam_dict in pid_to_cams.items():
            if pid in pid_to_3d:
                kps_3d = pid_to_3d[pid]
                if self._is_valid_skeleton(kps_3d):
                    valid_matched.append((pid, kps_3d, cam_dict))
                else:
                    invalid_matched.append((pid, cam_dict))
            else:
                for cam_idx, kps_xy in cam_dict.items():
                    scores = kps_xy[:, 2]
                    conf = float(scores[scores > 0].mean()) if (scores > 0).any() else 0.0
                    unmatched.append((conf, cam_idx, kps_xy))

        # Sort unmatched by confidence and limit
        unmatched.sort(key=lambda x: x[0], reverse=True)
        num_matched = len(valid_matched) + len(invalid_matched)
        slots_for_unmatched = max(0, self.max_people - num_matched) if not self.filter_unmatched else 0
        selected_unmatched = unmatched[:slots_for_unmatched]

        # Compute total slots needed
        if selected_unmatched:
            unmatched_per_view = np.zeros(self.num_cams, dtype=np.int32)
            for _, cam_idx, _ in selected_unmatched:
                unmatched_per_view[cam_idx] += 1
            total_people = num_matched + int(unmatched_per_view.max())
        else:
            total_people = num_matched

        if total_people == 0:
            return None, global_id_counter

        # Allocate output arrays
        gt_keypoints_xyzs = np.zeros((total_people, self.num_joints, 4), dtype=np.float32)
        gt_keypoints_xys = np.zeros((self.num_cams, total_people, self.num_joints, 3), dtype=np.float32)
        person_ids = np.full((self.num_cams, total_people), -1, dtype=np.int64)

        # Fill valid matched
        idx = 0
        for pid, kps_3d, cam_dict in valid_matched:
            gt_keypoints_xyzs[idx] = kps_3d
            for cam_idx, kps_xy in cam_dict.items():
                person_ids[cam_idx, idx] = pid
                gt_keypoints_xys[cam_idx, idx] = kps_xy
            idx += 1

        # Fill invalid matched (no person_ids, no 3D)
        for _, cam_dict in invalid_matched:
            for cam_idx, kps_xy in cam_dict.items():
                gt_keypoints_xys[cam_idx, idx] = kps_xy
            idx += 1

        # Fill unmatched (per-view slots starting after matched)
        if selected_unmatched:
            next_slot = np.full(self.num_cams, idx, dtype=np.int32)
            next_pid = max((pid for pid, _, _ in valid_matched), default=-1) + 1
            for i, (_, cam_idx, kps_xy) in enumerate(selected_unmatched):
                slot = next_slot[cam_idx]
                person_ids[cam_idx, slot] = next_pid + i
                gt_keypoints_xys[cam_idx, slot] = kps_xy
                next_slot[cam_idx] += 1

        # Build image paths
        img_paths = []
        for cam_key in self.cam_list:
            img_path = self.data_root / sequence / "colorimage" / f"camera{cam_key:02d}_colorimage-{frame_num:06d}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"The path: {img_path} does not exist.")
            img_paths.append(img_path)

        return {
            "sequence": sequence,
            "frame_num": frame_num,
            "gt_keypoints_xyzs": gt_keypoints_xyzs,
            "gt_keypoints_xys": gt_keypoints_xys,
            "person_ids": person_ids,
            "global_ids": np.arange(global_id_counter, global_id_counter + total_people),
            "img_paths": img_paths,
            "cam_params_vec": cam_params,
        }, global_id_counter + total_people

    def __call__(self) -> tuple[dict, list[tuple]]:
        """Load and filter pseudo ground truth data."""
        sequences_data = {}
        frame_map = []
        global_id_counter = 0

        with open(self.label_file, "rb") as f:
            all_pseudo_data = pickle.load(f)

        for sequence in self.sequences:
            if sequence not in all_pseudo_data:
                continue

            seq_data = all_pseudo_data[sequence]
            cam_params = self.cam_params_per_sequence[sequence]

            available_frames = sorted([int(k) for k in seq_data.keys()])
            available_frames = available_frames[:: self.interval]

            current_sequence_frames = []

            rank = get_rank()
            pbar = tqdm(total=len(available_frames), desc=f"Loading {sequence} (Pseudo)...") if rank == 0 else NoOp()

            for frame_num in available_frames:
                if self.frame_interval is not None:
                    if frame_num not in range(self.frame_interval[0], self.frame_interval[1] + 1):
                        pbar.update(1)
                        continue

                frame_2d_data = seq_data[frame_num].get("2D")
                frame_3d_data = seq_data[frame_num].get("3D")

                if not frame_2d_data:
                    pbar.update(1)
                    continue

                frame_data, global_id_counter = self._process_frame(
                    sequence=sequence,
                    frame_num=frame_num,
                    frame_2d_data=frame_2d_data,
                    frame_3d_data=frame_3d_data,
                    cam_params=cam_params,
                    global_id_counter=global_id_counter,
                )

                if frame_data is not None:
                    current_sequence_frames.append(frame_data)

                pbar.update(1)

            pbar.close()

            if not current_sequence_frames:
                continue

            sequences_data[sequence] = current_sequence_frames

            for frame_idx_in_seq in range(len(current_sequence_frames)):
                frame_map.append((sequence, frame_idx_in_seq))

        return sequences_data, frame_map


def get_cam_params(sequences: list[str], data_root: Path, cameras: list[int]) -> dict[str, np.ndarray]:
    cam_params = {}

    # 1. Coordinate Transforms (Flip/Swap)
    T_flip = np.eye(4, dtype=np.float32)
    T_flip[:3, :3] = R.from_euler("x", 180, degrees=True).as_matrix()

    T_swap = np.eye(4, dtype=np.float32)
    T_swap[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix()

    # 2. Unit Scaling (Meters -> Millimeters)
    # Note: Even if you want mm, apply this consistently to ALL translations.
    SCALE_M_TO_MM = 1000.0

    for sequence in sequences:
        sequence_path = data_root / sequence
        cam_params_seq = []

        for cam_idx in sorted(cameras):
            cal_path = sequence_path / f"camera{cam_idx:02d}.json"
            with open(cal_path, "r") as f:
                data = json.load(f)["value0"]

            # --- A. Load Base Pose (Depth Sensor to World) ---
            t_dict = data["camera_pose"]["translation"]
            # Apply scaling immediately
            t_raw = np.array([t_dict["m00"], t_dict["m10"], t_dict["m20"]], dtype=np.float32) * SCALE_M_TO_MM

            q_raw = [data["camera_pose"]["rotation"][k] for k in ["x", "y", "z", "w"]]
            r_raw = R.from_quat(q_raw).as_matrix().astype(np.float32)

            T_depth_to_world = np.eye(4, dtype=np.float32)
            T_depth_to_world[:3, :3] = r_raw
            T_depth_to_world[:3, 3] = t_raw

            # --- B. Load Color-to-Depth Transform ---
            # Maps from the color frame to the depth frame (key 'color2depth_transform').
            c2d_data = data.get("color2depth_transform", None)

            if c2d_data:
                t_c2d_dict = c2d_data["translation"]
                # Ensure this is ALSO scaled to MM
                t_c2d = (
                    np.array([t_c2d_dict["m00"], t_c2d_dict["m10"], t_c2d_dict["m20"]], dtype=np.float32)
                    * SCALE_M_TO_MM
                )

                q_c2d = [
                    c2d_data["rotation"]["x"],
                    c2d_data["rotation"]["y"],
                    c2d_data["rotation"]["z"],
                    c2d_data["rotation"]["w"],
                ]
                r_c2d = R.from_quat(q_c2d).as_matrix().astype(np.float32)

                T_color_to_depth = np.eye(4, dtype=np.float32)
                T_color_to_depth[:3, :3] = r_c2d
                T_color_to_depth[:3, 3] = t_c2d

                # Compose World <- Depth <- Color so that
                # P_world = T_depth_to_world @ T_color_to_depth @ P_color
                T_raw_color = T_depth_to_world @ T_color_to_depth
            else:
                # If absent, assume the color and depth frames coincide.
                T_raw_color = T_depth_to_world

            # --- C. Apply Coordinate System Fixes (Swap/Flip) ---
            # T_c2w: The final Color Camera Pose in World Space
            T_c2w = T_swap @ T_raw_color @ T_flip

            # --- D. Invert for View Matrix (World to Camera) ---
            T_w2c = np.linalg.inv(T_c2w)
            R_w2c = T_w2c[:3, :3]
            t_w2c = T_w2c[:3, 3]

            # Extract C2W components
            R_c2w = T_c2w[:3, :3]
            t_c2w = T_c2w[:3, 3]

            # --- E. Intrinsics & Distortion ---
            i_dict = data["color_parameters"]["intrinsics_matrix"]
            K = np.array(
                [
                    [i_dict["m00"], i_dict["m10"], i_dict["m20"]],
                    [i_dict["m01"], i_dict["m11"], i_dict["m21"]],
                    [i_dict["m02"], i_dict["m12"], i_dict["m22"]],
                ],
                dtype=np.float32,
            )
            K[2, 1], K[2, 2] = 0.0, 1.0
            K_inv = np.linalg.inv(K)

            rad_dict = data["color_parameters"]["radial_distortion"]
            tan_dict = data["color_parameters"]["tangential_distortion"]
            k = np.array(
                [
                    rad_dict.get("k1", 0),
                    rad_dict.get("k2", 0),
                    rad_dict.get("k3", 0),
                    rad_dict.get("k4", 0),
                    rad_dict.get("k5", 0),
                    rad_dict.get("k6", 0),
                ],
                dtype=np.float32,
            )
            p = np.array([tan_dict.get("p1", 0), tan_dict.get("p2", 0)], dtype=np.float32)

            # --- F. Pack ---
            cam_params_v = np.hstack(
                [
                    R_w2c.flatten(),
                    R_c2w.flatten(),
                    t_w2c.flatten(),
                    t_c2w.flatten(),
                    K.flatten(),
                    K_inv.flatten(),
                    k.flatten(),
                    p.flatten(),
                    np.array([cam_idx], dtype=np.float32),
                ]
            )
            cam_params_seq.append(cam_params_v)

        cam_params[sequence] = np.stack(cam_params_seq)

    return cam_params
