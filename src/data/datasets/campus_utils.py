"""
Utility functions for the Campus dataset
"""

import json
import pickle
from pathlib import Path

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from src.utils.camera import world_3d_to_img_2d
from src.utils.common import NoOp, get_rank


class GTDataLoader:
    """Loader for ground truth data from Campus dataset."""

    def __init__(
        self,
        cam_params_per_sequence: dict[str, np.ndarray],
        cam_list: list[int],
        data_root: Path,
        sequences: list[str],
        interval: int = 1,
        frame_interval: list[int] | None = None,
        root_idx: int = 2,
    ):
        self.cam_params_per_sequence = cam_params_per_sequence
        self.cam_list = cam_list
        self.data_root = data_root
        self.sequences = sequences  # For Campus, this is always ["defalta"]
        self.interval = interval
        self.frame_interval = frame_interval
        self.root_idx = root_idx

        self.num_joints = 14  # Campus convention

    def _get_frame_list(self) -> list[int]:
        """Get list of frames to process based on frame_interval."""
        if self.frame_interval is not None:
            return self.frame_interval
        else:
            # Default Campus frame range (full dataset)
            return list(range(0, 2000))

    def _process_frame(
        self,
        sequence: str,
        frame_num: int,
        actor_3d: np.ndarray,
        cam_params: np.ndarray,
        global_id_counter: int,
    ) -> tuple[dict | None, int]:
        """Process a single frame and return frame data."""
        num_person = len(actor_3d)

        gt_keypoints_xyzs, person_ids, global_ids = [], [], []

        for person in range(num_person):
            pose3d = actor_3d[person][frame_num] * 1000.0  # Convert to mm

            # Skip if no valid pose data
            if len(pose3d[0]) == 0:
                continue

            # pose3d is (14, 3) - add visibility score column
            joints_3d = np.zeros((self.num_joints, 4), dtype=np.float32)
            joints_3d[:, :3] = pose3d
            joints_3d[:, 3] = 1.0  # All joints visible in GT

            gt_keypoints_xyzs.append(joints_3d)
            person_ids.append(person)
            global_ids.append(global_id_counter)
            global_id_counter += 1

        if not gt_keypoints_xyzs:
            return None, global_id_counter

        gt_keypoints_xyzs = np.array(gt_keypoints_xyzs)
        person_ids = np.array(person_ids)
        global_ids = np.array(global_ids)

        # Person IDs are scene-global; broadcast across views to match
        # MMORDataset / ShelfDataset shape (V, N) so the downstream code that
        # indexes by (B, A, T, V, N) sees the right rank.
        person_ids = person_ids[None].repeat(cam_params.shape[0], 0)

        # Project into Multi-View 2D
        gt_keypoints_xy, gt_keypoints_valid = world_3d_to_img_2d(
            torch.from_numpy(gt_keypoints_xyzs[..., :3]),
            torch.from_numpy(cam_params),
        )
        gt_keypoints_xy = gt_keypoints_xy.numpy()
        gt_keypoints_valid = gt_keypoints_valid.numpy()

        # gt_keypoints_s is (NumCams, NumPeople, 14, 1)
        gt_keypoints_s = gt_keypoints_xyzs[None, ..., -1:]
        gt_keypoints_s = gt_keypoints_s.repeat(cam_params.shape[0], 0)

        # Concatenate xy with scores
        gt_keypoints_xys = np.concatenate([gt_keypoints_xy, gt_keypoints_s * gt_keypoints_valid], axis=-1)

        # Image paths
        img_paths = []
        for cam_idx in self.cam_list:
            img_path = self.data_root / f"Camera{cam_idx}" / f"campus4-c{cam_idx}-{frame_num:05d}.png"
            if not img_path.exists():
                raise FileNotFoundError(f"The path: {img_path} does not exist.")
            img_paths.append(img_path)

        frame_data = {
            "sequence": sequence,
            "frame_num": frame_num,
            "gt_keypoints_xyzs": gt_keypoints_xyzs,
            "gt_keypoints_xys": gt_keypoints_xys,
            "person_ids": person_ids,
            "global_ids": global_ids,
            "img_paths": img_paths,
            "cam_params_vec": cam_params,
        }

        return frame_data, global_id_counter

    def __call__(self) -> tuple[dict, list[tuple]]:
        """Load ground truth data from actorsGT.mat file."""
        sequences_data = {}
        frame_map = []
        global_id_counter = 0

        # Load ground truth from MAT file
        datafile = self.data_root / "actorsGT.mat"
        actor_3d = scipy.io.loadmat(datafile)["actor3D"]
        actor_3d = np.array(np.array(actor_3d.tolist()).tolist(), dtype=object).squeeze()

        # Campus has single sequence "defalta"
        sequence = "defalta"
        cam_params = self.cam_params_per_sequence[sequence]

        frame_list = self._get_frame_list()
        frame_list = frame_list[:: self.interval]

        current_sequence_frames = []

        rank = get_rank()
        pbar = tqdm(total=len(frame_list), desc=f"Loading {sequence}...") if rank == 0 else NoOp()

        for frame_num in frame_list:
            frame_data, global_id_counter = self._process_frame(
                sequence=sequence,
                frame_num=frame_num,
                actor_3d=actor_3d,
                cam_params=cam_params,
                global_id_counter=global_id_counter,
            )

            if frame_data is not None:
                current_sequence_frames.append(frame_data)

            pbar.update(1)

        pbar.close()

        if current_sequence_frames:
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
        frame_interval: list[int] | None = None,
        root_idx: int = 2,
        max_people: int = 10,
    ):
        self.label_file = Path(label_file)
        self.cam_params_per_sequence = cam_params_per_sequence
        self.cam_list = cam_list
        self.data_root = data_root
        self.sequences = sequences  # For Campus, this is always ["defalta"]
        self.interval = interval
        self.frame_interval = frame_interval if frame_interval is not None else []
        self.root_idx = root_idx
        self.min_visible_joints = min_visible_joints
        self.max_people = max_people

        self.num_joints = 15  # Pseudo labels are in Panoptic convention
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

        # Camera keys for Campus are integers
        cam_keys = self.cam_list

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

        for cam_idx, cam_key in enumerate(cam_keys):
            if str(cam_key) not in frame_2d_data:
                continue
            data_src = frame_2d_data[str(cam_key)]
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
                kps_3d = triangulated_xyzs[idx_3d]  # type: ignore
                if self._is_valid_skeleton(kps_3d):
                    valid_matched.append((pid, kps_3d, cam_dict))
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
        person_ids = np.full(total_people, -1, dtype=np.int64)

        idx = 0

        # Valid matched
        for pid, kps_3d, cam_dict in valid_matched:
            person_ids[idx] = pid
            gt_keypoints_xyzs[idx] = kps_3d
            for cam_idx, kps_xy in cam_dict.items():
                gt_keypoints_xys[cam_idx, idx] = kps_xy
            idx += 1

        # Invalid matched
        for pid, cam_dict in invalid_matched:
            for cam_idx, kps_xy in cam_dict.items():
                gt_keypoints_xys[cam_idx, idx] = kps_xy
            idx += 1

        # Unmatched
        for conf, cam_idx, kps_xy in selected_unmatched:
            gt_keypoints_xys[cam_idx, idx] = kps_xy
            idx += 1

        # Vectorized joint score computation for valid matched
        if num_valid > 0:
            valid_2d = gt_keypoints_xys[:, :num_valid]  # (C, V, J, 3)
            scores_2d = valid_2d[:, :, :, 2]

            xy = valid_2d[:, :, :, :2]

            valid_mask = (scores_2d > 0) & ~((xy[:, :, :, 0] == 0) & (xy[:, :, :, 1] == 0))

            masked_scores = np.where(valid_mask, scores_2d, np.nan)
            with np.errstate(all="ignore"):
                mean_scores = np.nanmean(masked_scores, axis=0)  # (V, J)

            mean_scores = np.nan_to_num(mean_scores, nan=0.0)

            # Ensure we don't assign high scores to 3D keypoints that are actually missing (0,0,0)
            valid_3d_coords = gt_keypoints_xyzs[:num_valid, :, :3]
            is_valid_joint_3d = ~np.all(valid_3d_coords == 0, axis=-1)  # (V, J)

            # Mask out scores where 3D is invalid
            final_scores = mean_scores * is_valid_joint_3d.astype(np.float32)

            gt_keypoints_xyzs[:num_valid, :, 3] = final_scores

        # Image paths for Campus
        img_paths = []
        for cam_idx in self.cam_list:
            img_path = self.data_root / f"Camera{cam_idx}" / f"campus4-c{cam_idx}-{frame_num:05d}.png"
            if not img_path.exists():
                raise FileNotFoundError(f"The path: {img_path} does not exist.")
            img_paths.append(img_path)

        # Person IDs are scene-global; broadcast across views to match
        # MMORDataset / ShelfDataset shape (V, N) so the downstream code that
        # indexes by (B, A, T, V, N) sees the right rank.
        person_ids = person_ids[None].repeat(self.num_cams, 0)

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

        # Campus has single sequence "defalta"
        sequence = "defalta"

        if sequence not in all_pseudo_data:
            return sequences_data, frame_map

        seq_data = all_pseudo_data[sequence]
        cam_params = self.cam_params_per_sequence[sequence]

        available_frames = sorted([int(k) for k in seq_data.keys()])
        available_frames = available_frames[:: self.interval]

        current_sequence_frames = []

        rank = get_rank()
        pbar = tqdm(total=len(available_frames), desc=f"Loading {sequence} (Pseudo)...") if rank == 0 else NoOp()

        for frame_num in available_frames:
            if frame_num not in self.frame_interval:
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

        if current_sequence_frames:
            sequences_data[sequence] = current_sequence_frames

            for frame_idx_in_seq in range(len(current_sequence_frames)):
                frame_map.append((sequence, frame_idx_in_seq))

        return sequences_data, frame_map


def get_cam_params(sequences: list[str], data_root: Path, cam_list: list[int]):
    """
    Initialize Camera Parameters for Campus dataset.

    Note: Campus uses a different camera parameter convention than Panoptic.
    - T in calibration file is the camera center (C), not translation vector
    - No coordinate transform matrix M is applied
    """
    cam_params = {}

    cal_path = data_root / "calibration_campus.json"
    with cal_path.open("r") as f:
        sequence_calibration = json.load(f)

    cam_params_seq = []
    for cam_idx in cam_list:
        cam_calib = sequence_calibration.get(str(cam_idx))

        if cam_calib is None:
            continue

        R_w2c = np.asarray(cam_calib["R"], dtype=np.float32)

        # T in Campus calibration is Camera Center (C), not translation vector (t)
        C_center = np.asarray(cam_calib["T"], dtype=np.float32)

        # Calculate the translation vector t = -R @ C
        t_w2c = -1.0 * (R_w2c @ C_center.reshape(3, 1)).flatten()

        # R_c2w is just the transpose (inverse of rotation matrix)
        R_c2w = R_w2c.T

        # t_c2w is the Camera Center (C)
        t_c2w = C_center

        fx = cam_calib["fx"]
        fy = cam_calib["fy"]
        cx = cam_calib["cx"]
        cy = cam_calib["cy"]

        K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        K_inv = np.linalg.inv(K)

        k = np.array(cam_calib["k"], dtype=np.float32).flatten()
        p = np.array(cam_calib["p"], dtype=np.float32).flatten()

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

    # Campus has single sequence "defalta"
    cam_params["defalta"] = np.stack(cam_params_seq)

    return cam_params
