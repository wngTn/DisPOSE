"""
Utility functions for the panoptic dataset
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.camera import world_3d_to_img_2d
from src.utils.common import NoOp, get_rank


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
    ):
        self.cam_params_per_sequence = cam_params_per_sequence
        self.cam_list = cam_list
        self.data_root = data_root
        self.sequences = sequences
        self.interval = interval
        self.frame_interval = frame_interval
        self.root_idx = root_idx

        # Rotation matrix for coordinate transform
        self.M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    def _process_frame(
        self,
        sequence: str,
        annotation_file: Path,
        cam_params: np.ndarray,
        global_id_counter: int,
    ) -> tuple[dict | None, int]:
        """Process a single annotation file and return frame data."""
        frame_num_str = annotation_file.stem.split("_")[-1]

        with annotation_file.open("r", encoding="utf-8") as f:
            annotation_3d = json.load(f)

        gt_keypoints_xyzs, person_ids, global_ids = [], [], []
        body_dicts = annotation_3d["bodies"]

        for body_dict in body_dicts:
            person_id = body_dict["id"]
            joints_3d = np.array(body_dict["joints19"], dtype=np.float32).reshape(19, 4)[:15]
            visible = joints_3d[:, -1] > 0.1
            joints_3d[~visible, -1] = 0.0

            if not visible[self.root_idx]:
                continue

            joints_3d[:, :3] = joints_3d[:, :3].dot(self.M) * 10.0

            gt_keypoints_xyzs.append(joints_3d)
            person_ids.append(person_id)
            global_ids.append(global_id_counter)
            global_id_counter += 1

        if not gt_keypoints_xyzs:
            return None, global_id_counter

        gt_keypoints_xyzs = np.array(gt_keypoints_xyzs)
        person_ids = np.array(person_ids)
        person_ids = person_ids[None].repeat(cam_params.shape[0], 0)
        global_ids = np.array(global_ids)

        # Project into Multi-View 2D
        gt_keypoints_xy, gt_keypoints_valid = world_3d_to_img_2d(
            torch.from_numpy(gt_keypoints_xyzs[..., :3]),
            torch.from_numpy(cam_params),
        )
        gt_keypoints_xy = gt_keypoints_xy.numpy()
        gt_keypoints_valid = gt_keypoints_valid.numpy()

        # gt_keypoints_s is (NumCams, NumPeople, 15, 1)
        gt_keypoints_s = gt_keypoints_xyzs[None, ..., -1:]
        gt_keypoints_s = gt_keypoints_s.repeat(cam_params.shape[0], 0)

        gt_keypoints_xys = np.concatenate([gt_keypoints_xy, gt_keypoints_s * gt_keypoints_valid], axis=-1)

        # Image paths
        img_paths = []
        for cam_id in self.cam_list:
            cam_view = f"{cam_id[0]:02d}_{cam_id[1]:02d}"
            img_path = self.data_root / sequence / "hdImgs" / cam_view / f"{cam_view}_{frame_num_str}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"The path: {img_path} does not exist.")
            img_paths.append(img_path)

        frame_data = {
            "sequence": sequence,
            "frame_num": int(frame_num_str),
            "gt_keypoints_xyzs": gt_keypoints_xyzs,
            "gt_keypoints_xys": gt_keypoints_xys,
            "person_ids": person_ids,
            "global_ids": global_ids,
            "img_paths": img_paths,
            "cam_params_vec": cam_params,
        }

        return frame_data, global_id_counter

    def __call__(self) -> tuple[dict, list[tuple]]:
        """Load ground truth data from annotation files."""
        sequences_data = {}
        frame_map = []
        global_id_counter = 0

        for sequence in self.sequences:
            annotation_dir = self.data_root / sequence / "hdPose3d_stage1_coco19"
            annotation_list = sorted(annotation_dir.iterdir())
            annotation_list = annotation_list[:: self.interval]

            if not annotation_list:
                continue

            cam_params = self.cam_params_per_sequence[sequence]
            current_sequence_frames = []

            rank = get_rank()
            pbar = tqdm(total=len(annotation_list), desc=f"Loading {sequence}...") if rank == 0 else NoOp()

            for annotation_file in annotation_list:
                frame_num_str = annotation_file.stem.split("_")[-1]

                if self.frame_interval is not None:
                    if int(frame_num_str) not in range(self.frame_interval[0], self.frame_interval[1] + 1):
                        pbar.update(1)
                        continue

                frame_data, global_id_counter = self._process_frame(
                    sequence=sequence,
                    annotation_file=annotation_file,
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


class PseudoGTDataLoader:
    """Loader for pseudo ground truth data with filtering capabilities."""

    def __init__(
        self,
        label_file: str,
        min_visible_joints: int,
        cam_params_per_sequence: dict[str, np.ndarray],
        cam_list: list[tuple],
        data_root: Path,
        sequences: list[str],
        interval: int,
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
        self.frame_interval = frame_interval
        self.root_idx = root_idx
        self.min_visible_joints = min_visible_joints
        self.max_people = max_people

        self.num_joints = 15
        self.num_cams = len(cam_list)

    def _is_valid_skeleton(self, keypoints_xyz: np.ndarray) -> bool:
        """Check if a 3D skeleton has enough valid joints."""
        valid_mask = ~np.all(keypoints_xyz == 0, axis=1)
        num_valid = np.sum(valid_mask)
        return num_valid >= self.min_visible_joints

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

        cam_keys = [f"{c[0]:02d}_{c[1]:02d}" for c in self.cam_list]

        if frame_3d_data and frame_3d_data["person_ids"].size:
            person_ids_3d = frame_3d_data["person_ids"]
            triangulated_xyzs = frame_3d_data["triangulated_keypoints_xyzs"]
            order = np.argsort(person_ids_3d)
            sorted_pids_3d = person_ids_3d[order]
            triangulated_xyzs = triangulated_xyzs[order]
            pid_to_3d_idx = {pid: i for i, pid in enumerate(sorted_pids_3d)}
        else:
            triangulated_xyzs = None
            pid_to_3d_idx = {}

        pid_to_cams: dict[int, dict[int, np.ndarray]] = {}

        for cam_idx, cam_key in enumerate([f"{c[0]:02d}_{c[1]:02d}" for c in self.cam_list]):
            if cam_key not in frame_2d_data:
                continue
            data_src = frame_2d_data[cam_key]
            src_kps = data_src["keypoints_xys"]
            src_ids = data_src["person_ids"]

            for i, pid in enumerate(src_ids):
                if pid not in pid_to_cams:
                    pid_to_cams[pid] = {}
                pid_to_cams[pid][cam_idx] = src_kps[i]

        if not pid_to_cams:
            return None, global_id_counter

        valid_matched = []
        invalid_matched = []
        unmatched_detections = []

        for pid, cam_dict in pid_to_cams.items():
            if pid in pid_to_3d_idx:
                idx_3d = pid_to_3d_idx[pid]
                kps_xyzs = triangulated_xyzs[idx_3d]  # type: ignore
                if self._is_valid_skeleton(kps_xyzs):
                    valid_matched.append((pid, kps_xyzs, cam_dict))
                else:
                    invalid_matched.append((pid, cam_dict))
            else:
                for cam_idx, kps_xy in cam_dict.items():
                    scores = kps_xy[:, 2]
                    valid_mask = scores > 0
                    conf = float(scores[valid_mask].mean()) if valid_mask.any() else 0.0
                    unmatched_detections.append((conf, cam_idx, kps_xy))

        unmatched_detections.sort(key=lambda x: x[0], reverse=True)

        num_valid = len(valid_matched)
        num_invalid = len(invalid_matched)
        num_matched = num_valid + num_invalid
        slots_for_unmatched = max(0, self.max_people - num_matched)
        selected_unmatched = unmatched_detections[:slots_for_unmatched]

        total_people = num_matched + len(selected_unmatched)
        if total_people == 0:
            return None, global_id_counter

        gt_keypoints_xyzs = np.zeros((total_people, self.num_joints, 4), dtype=np.float32)
        gt_keypoints_xys = np.zeros((self.num_cams, total_people, self.num_joints, 3), dtype=np.float32)
        person_ids = np.full((self.num_cams, total_people), -1, dtype=np.int64)

        idx = 0

        # Valid matched
        for pid, kps_xyzs, cam_dict in valid_matched:
            gt_keypoints_xyzs[idx] = kps_xyzs
            for cam_idx, kps_xy in cam_dict.items():
                person_ids[cam_idx, idx] = pid
                gt_keypoints_xys[cam_idx, idx] = kps_xy
            idx += 1

        # Invalid matched
        for pid, cam_dict in invalid_matched:
            for cam_idx, kps_xy in cam_dict.items():
                gt_keypoints_xys[cam_idx, idx] = kps_xy
            idx += 1

        # Unmatched (no person_ids assigned)
        for conf, cam_idx, kps_xy in selected_unmatched:
            gt_keypoints_xys[cam_idx, idx] = kps_xy
            idx += 1

        img_paths = []
        for cam_key in cam_keys:
            img_path = self.data_root / sequence / "hdImgs" / cam_key / f"{cam_key}_{frame_num:08d}.jpg"
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


def get_cam_params(sequences: list[str], data_root: Path, cam_list: list[tuple]):
    """
    Initialize Camera Parameters
    """
    cam_params = {}
    for sequence in sequences:
        cam_params[sequence] = []

        sequence_calibration_file = data_root / sequence / f"calibration_{sequence}.json"
        with sequence_calibration_file.open("r") as f:
            sequence_calibration = json.load(f)

        cam_params_seq = []
        for cam_idx, cam_view in enumerate(cam_list):
            cam_calib = next(
                (x for x in sequence_calibration["cameras"] if x["name"] == f"{cam_view[0]:02d}_{cam_view[1]:02d}"),
                None,
            )
            if cam_calib is None:
                continue

            M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            R_w2c_raw = np.asarray(cam_calib["R"], dtype=np.float32)
            t_w2c_raw = np.asarray(cam_calib["t"], dtype=np.float32)

            # --- Our Cameras Are in Column VECTOR CONVENTION ---
            R_w2c = R_w2c_raw @ M
            R_c2w = R_w2c.T

            # t_w2c (translation) is unchanged. It's the 3-element column vector.
            t_w2c = t_w2c_raw.copy()
            t_w2c *= 10.0

            t_c2w = -(R_w2c.T @ t_w2c)

            K = np.array(cam_calib["K"])
            K_inv = np.linalg.inv(K)

            k = np.array(cam_calib["distCoef"])[[0, 1, 4]]
            p = np.array(cam_calib["distCoef"])[[2, 3]]

            # Pad k to length 6
            if len(k) < 6:
                k = np.pad(k, (0, 6 - len(k)), mode="constant", constant_values=0.0)

            # Build our camera vector
            # 9 + 9 + 3 + 3 + 9 + 9 + 6 + 2 + 1
            # R_w2c + R_c2w + T_w2c + T_c2w + K + K_inv + k + p + cam_idx
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
                    cam_idx,
                ],
                dtype=np.float32,
            )
            cam_params_seq.append(cam_params_v)

        cam_params[sequence] = np.stack(cam_params_seq)

    return cam_params
