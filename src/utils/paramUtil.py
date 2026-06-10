"""Dataset convention tables and parameter utilities.

Defines per-dataset skeleton topologies (JOINT_CONVENTIONS), keypoint and bone
metadata (KEYPOINT_INFO, SKELETON_INFO), camera configurations, person-ID color
mapping, and panoptic-to-Campus/Shelf joint conversions.
"""

import numpy as np
import torch


# Colorblind-friendly palette (RGB, 0-1 range)
# Based on Wong (2011) and IBM Design Language, optimized for deuteranopia/protanopia
PERSON_PALETTE = np.array(
    [
        [0.000, 0.447, 0.698],  # Blue
        [0.902, 0.624, 0.000],  # Orange
        [0.000, 0.620, 0.451],  # Teal
        [0.835, 0.369, 0.000],  # Vermillion
        [0.573, 0.467, 0.800],  # Purple
        [0.337, 0.706, 0.914],  # Sky blue
        [0.459, 0.439, 0.702],  # Indigo
        [0.941, 0.894, 0.259],  # Yellow
        [0.651, 0.463, 0.114],  # Brown
    ],
    dtype=np.float32,
)


def person_id_to_color(
    person_idx: int,
    alpha: float = 1.0,
    as_uint8: bool = False,
) -> np.ndarray:
    """
    Map a person index to a consistent, colorblind-friendly color.

    Args:
        person_idx: Integer index for the person (0-indexed, wraps for idx >= 10).
        alpha: Opacity value in [0, 1].
        as_uint8: If True, return RGBA as uint8 (0-255); else float (0-1).

    Returns:
        RGBA array of shape (4,).
    """
    rgb = PERSON_PALETTE[person_idx % len(PERSON_PALETTE)]
    rgba = np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.float32)

    if as_uint8:
        return (rgba * 255).astype(np.uint8)
    return rgba


JOINT_CONVENTIONS: dict[str, list[tuple[int, int]]] = {
    "panoptic": [
        (0, 1),
        (0, 2),
        (0, 3),
        (3, 4),
        (4, 5),
        (0, 9),
        (9, 10),
        (10, 11),
        (2, 6),
        (2, 12),
        (6, 7),
        (7, 8),
        (12, 13),
        (13, 14),
    ],
    "coco": [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (11, 13),
        (13, 15),
        (6, 12),
        (12, 14),
        (14, 16),
        (5, 6),
        (11, 12),
    ],
    "shelf": [
        (1, 0),  # right fibula
        (2, 1),  # right femur
        (3, 4),  # left femur
        (4, 5),  # left fibula
        (3, 2),  # mid-hip joint
        (7, 6),  # right ulna
        (8, 7),  # right humerus
        (9, 10),  # left humerus
        (10, 11),  # left ulna
        (8, 2),  # right torso
        (9, 3),  # left torso
        (12, 8),  # right shoulder
        (12, 9),  # left shoulder
        (13, 12),  # head
    ],
    "campus": [
        (1, 0),  # right fibula
        (2, 1),  # right femur
        (3, 4),  # left femur
        (4, 5),  # left fibula
        (3, 2),  # mid-hip joint
        (7, 6),  # right ulna
        (8, 7),  # right humerus
        (9, 10),  # left humerus
        (10, 11),  # left ulna
        (8, 2),  # right torso
        (9, 3),  # left torso
        (12, 8),  # right shoulder
        (12, 9),  # left shoulder
        (13, 12),  # head
    ],
}

KEYPOINT_INFO = {
    "panoptic": {
        0: dict(name="neck", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(name="nose", id=1, color=[51, 153, 255], type="upper", swap=""),
        2: dict(name="mid_hip", id=2, color=[0, 255, 0], type="lower", swap=""),
        3: dict(name="left_shoulder", id=3, color=[0, 255, 0], type="upper", swap="right_shoulder"),
        4: dict(name="left_elbow", id=4, color=[0, 255, 0], type="upper", swap="right_elbow"),
        5: dict(name="left_wrist", id=5, color=[0, 255, 0], type="upper", swap="right_wrist"),
        6: dict(name="left_hip", id=6, color=[0, 255, 0], type="lower", swap="right_hip"),
        7: dict(name="left_knee", id=7, color=[0, 255, 0], type="lower", swap="right_knee"),
        8: dict(name="left_ankle", id=8, color=[0, 255, 0], type="lower", swap="right_ankle"),
        9: dict(name="right_shoulder", id=9, color=[255, 128, 0], type="upper", swap="left_shoulder"),
        10: dict(name="right_elbow", id=10, color=[255, 128, 0], type="upper", swap="left_elbow"),
        11: dict(name="right_wrist", id=11, color=[255, 128, 0], type="upper", swap="left_wrist"),
        12: dict(name="right_hip", id=12, color=[255, 128, 0], type="lower", swap="left_hip"),
        13: dict(name="right_knee", id=13, color=[255, 128, 0], type="lower", swap="left_knee"),
        14: dict(name="right_ankle", id=14, color=[255, 128, 0], type="lower", swap="left_ankle"),
    },
    "coco": {
        0: dict(name="nose", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(name="left_eye", id=1, color=[51, 153, 255], type="upper", swap="right_eye"),
        2: dict(name="right_eye", id=2, color=[51, 153, 255], type="upper", swap="left_eye"),
        3: dict(name="left_ear", id=3, color=[51, 153, 255], type="upper", swap="right_ear"),
        4: dict(name="right_ear", id=4, color=[51, 153, 255], type="upper", swap="left_ear"),
        5: dict(name="left_shoulder", id=5, color=[0, 255, 0], type="upper", swap="right_shoulder"),
        6: dict(name="right_shoulder", id=6, color=[255, 128, 0], type="upper", swap="left_shoulder"),
        7: dict(name="left_elbow", id=7, color=[0, 255, 0], type="upper", swap="right_elbow"),
        8: dict(name="right_elbow", id=8, color=[255, 128, 0], type="upper", swap="left_elbow"),
        9: dict(name="left_wrist", id=9, color=[0, 255, 0], type="upper", swap="right_wrist"),
        10: dict(name="right_wrist", id=10, color=[255, 128, 0], type="upper", swap="left_wrist"),
        11: dict(name="left_hip", id=11, color=[0, 255, 0], type="lower", swap="right_hip"),
        12: dict(name="right_hip", id=12, color=[255, 128, 0], type="lower", swap="left_hip"),
        13: dict(name="left_knee", id=13, color=[0, 255, 0], type="lower", swap="right_knee"),
        14: dict(name="right_knee", id=14, color=[255, 128, 0], type="lower", swap="left_knee"),
        15: dict(name="left_ankle", id=15, color=[0, 255, 0], type="lower", swap="right_ankle"),
        16: dict(name="right_ankle", id=16, color=[255, 128, 0], type="lower", swap="left_ankle"),
    },
    "campus": {
        0: dict(name="right_ankle", id=0, color=[255, 128, 0], type="lower", swap="left_ankle"),
        1: dict(name="right_knee", id=1, color=[255, 128, 0], type="lower", swap="left_knee"),
        2: dict(name="right_hip", id=2, color=[255, 128, 0], type="lower", swap="left_hip"),
        3: dict(name="left_hip", id=3, color=[0, 255, 0], type="lower", swap="right_hip"),
        4: dict(name="left_knee", id=4, color=[0, 255, 0], type="lower", swap="right_knee"),
        5: dict(name="left_ankle", id=5, color=[0, 255, 0], type="lower", swap="right_ankle"),
        6: dict(name="right_wrist", id=6, color=[255, 128, 0], type="upper", swap="left_wrist"),
        7: dict(name="right_elbow", id=7, color=[255, 128, 0], type="upper", swap="left_elbow"),
        8: dict(name="right_shoulder", id=8, color=[255, 128, 0], type="upper", swap="left_shoulder"),
        9: dict(name="left_shoulder", id=9, color=[0, 255, 0], type="upper", swap="right_shoulder"),
        10: dict(name="left_elbow", id=10, color=[0, 255, 0], type="upper", swap="right_elbow"),
        11: dict(name="left_wrist", id=11, color=[0, 255, 0], type="upper", swap="right_wrist"),
        12: dict(name="bottom_head", id=12, color=[51, 153, 255], type="upper", swap=""),
        13: dict(name="top_head", id=13, color=[51, 153, 255], type="upper", swap=""),
    },
    "shelf": {
        0: dict(name="right_ankle", id=0, color=[255, 128, 0], type="lower", swap="left_ankle"),
        1: dict(name="right_knee", id=1, color=[255, 128, 0], type="lower", swap="left_knee"),
        2: dict(name="right_hip", id=2, color=[255, 128, 0], type="lower", swap="left_hip"),
        3: dict(name="left_hip", id=3, color=[0, 255, 0], type="lower", swap="right_hip"),
        4: dict(name="left_knee", id=4, color=[0, 255, 0], type="lower", swap="right_knee"),
        5: dict(name="left_ankle", id=5, color=[0, 255, 0], type="lower", swap="right_ankle"),
        6: dict(name="right_wrist", id=6, color=[255, 128, 0], type="upper", swap="left_wrist"),
        7: dict(name="right_elbow", id=7, color=[255, 128, 0], type="upper", swap="left_elbow"),
        8: dict(name="right_shoulder", id=8, color=[255, 128, 0], type="upper", swap="left_shoulder"),
        9: dict(name="left_shoulder", id=9, color=[0, 255, 0], type="upper", swap="right_shoulder"),
        10: dict(name="left_elbow", id=10, color=[0, 255, 0], type="upper", swap="right_elbow"),
        11: dict(name="left_wrist", id=11, color=[0, 255, 0], type="upper", swap="right_wrist"),
        12: dict(name="bottom_head", id=12, color=[51, 153, 255], type="upper", swap=""),
        13: dict(name="top_head", id=13, color=[51, 153, 255], type="upper", swap=""),
    },
}

SKELETON_INFO = {
    "panoptic": {
        # --- Head & Neck ---
        # (0, 1) | 239.0994 | 21.4246
        0: dict(link=("nose", "neck"), id=0, color=[51, 153, 255], mean=239.0994, std=21.4246),
        # --- Shoulders (Clavicles) ---
        # std widened to 45mm to reflect the higher annotation variance on the clavicle links.
        1: dict(link=("neck", "left_shoulder"), id=1, color=[0, 255, 0], mean=163.7494, std=45.0000),
        2: dict(link=("neck", "right_shoulder"), id=2, color=[255, 128, 0], mean=163.6601, std=45.0000),
        # --- Arms ---
        # (3, 4) | 270.2705 | 22.1400
        3: dict(link=("left_shoulder", "left_elbow"), id=3, color=[0, 255, 0], mean=270.2705, std=22.1400),
        # (9, 10) | 275.8651 | 23.2205
        4: dict(link=("right_shoulder", "right_elbow"), id=4, color=[255, 128, 0], mean=275.8651, std=23.2205),
        # (4, 5) | 234.7036 | 28.8417
        5: dict(link=("left_elbow", "left_wrist"), id=5, color=[0, 255, 0], mean=234.7036, std=28.8417),
        # (10, 11) | 236.2475 | 24.5372
        6: dict(link=("right_elbow", "right_wrist"), id=6, color=[255, 128, 0], mean=236.2475, std=24.5372),
        # --- Legs ---
        # (7, 8) | 381.2065 | 36.0775
        7: dict(link=("left_ankle", "left_knee"), id=7, color=[0, 255, 0], mean=381.2065, std=36.0775),
        # (6, 7) | 383.4786 | 25.0108
        8: dict(link=("left_knee", "left_hip"), id=8, color=[0, 255, 0], mean=383.4786, std=25.0108),
        # (13, 14) | 381.6285 | 33.2953
        9: dict(link=("right_ankle", "right_knee"), id=9, color=[255, 128, 0], mean=381.6285, std=33.2953),
        # (12, 13) | 388.6202 | 30.0957
        10: dict(link=("right_knee", "right_hip"), id=10, color=[255, 128, 0], mean=388.6202, std=30.0957),
        # --- Hips & Torso ---
        # (2, 6) | 104.0389 | 8.6003
        11: dict(link=("mid_hip", "left_hip"), id=11, color=[0, 255, 0], mean=104.0389, std=8.6003),
        # (2, 12) | 104.1686 | 17.0171
        12: dict(link=("mid_hip", "right_hip"), id=12, color=[255, 128, 0], mean=104.1686, std=17.0171),
        # --- Spine ---
        # std widened to 80mm to reflect the higher annotation variance on the spine link.
        13: dict(link=("mid_hip", "neck"), id=13, color=[51, 153, 255], mean=507.4331, std=80.0000),
    },
    "campus": {
        # 0: right_ankle -> right_knee | Table Index (1, 0)
        0: dict(link=("right_ankle", "right_knee"), id=0, color=[255, 128, 0], mean=455.7302, std=29.7887),
        # 1: right_knee -> right_hip | Table Index (2, 1)
        1: dict(link=("right_knee", "right_hip"), id=1, color=[255, 128, 0], mean=448.6274, std=43.7518),
        # 2: left_hip -> left_knee | Table Index (3, 4)
        2: dict(link=("left_hip", "left_knee"), id=2, color=[0, 255, 0], mean=439.2921, std=42.3366),
        # 3: left_knee -> left_ankle | Table Index (4, 5)
        3: dict(link=("left_knee", "left_ankle"), id=3, color=[0, 255, 0], mean=468.9776, std=36.0622),
        # 4: right_hip -> left_hip | Table Index (3, 2)
        4: dict(link=("right_hip", "left_hip"), id=4, color=[51, 153, 255], mean=213.1888, std=28.7003),
        # 5: right_wrist -> right_elbow | Table Index (7, 6)
        5: dict(link=("right_wrist", "right_elbow"), id=5, color=[255, 128, 0], mean=299.0551, std=64.7242),
        # 6: right_elbow -> right_shoulder | Table Index (8, 7)
        6: dict(link=("right_elbow", "right_shoulder"), id=6, color=[255, 128, 0], mean=334.2639, std=39.1762),
        # 7: left_shoulder -> left_elbow | Table Index (9, 10)
        7: dict(link=("left_shoulder", "left_elbow"), id=7, color=[0, 255, 0], mean=328.2070, std=47.0259),
        # 8: left_elbow -> left_wrist | Table Index (10, 11)
        8: dict(link=("left_elbow", "left_wrist"), id=8, color=[0, 255, 0], mean=288.8403, std=56.9613),
        # 9: right_hip -> right_shoulder | Table Index (8, 2)
        9: dict(link=("right_hip", "right_shoulder"), id=9, color=[255, 128, 0], mean=652.3549, std=69.0225),
        # 10: left_hip -> left_shoulder | Table Index (9, 3)
        10: dict(link=("left_hip", "left_shoulder"), id=10, color=[0, 255, 0], mean=657.7887, std=70.4995),
        # 11: right_shoulder -> bottom_head | Table Index (12, 8)
        11: dict(link=("right_shoulder", "bottom_head"), id=11, color=[255, 128, 0], mean=189.6254, std=24.2607),
        # 12: left_shoulder -> bottom_head | Table Index (12, 9)
        12: dict(link=("left_shoulder", "bottom_head"), id=12, color=[0, 255, 0], mean=197.3906, std=17.4256),
        # 13: bottom_head -> top_head | Table Index (13, 12)
        13: dict(link=("bottom_head", "top_head"), id=13, color=[51, 153, 255], mean=265.5409, std=26.8756),
    },
    "shelf": {
        # 0: right_ankle -> right_knee | Table Index (1, 0)
        0: dict(link=("right_ankle", "right_knee"), id=0, color=[255, 128, 0], mean=389.6704, std=28.8285),
        # 1: right_knee -> right_hip | Table Index (2, 1)
        1: dict(link=("right_knee", "right_hip"), id=1, color=[255, 128, 0], mean=381.0596, std=29.2054),
        # 2: left_hip -> left_knee | Table Index (3, 4)
        2: dict(link=("left_hip", "left_knee"), id=2, color=[0, 255, 0], mean=387.7613, std=27.1697),
        # 3: left_knee -> left_ankle | Table Index (4, 5)
        3: dict(link=("left_knee", "left_ankle"), id=3, color=[0, 255, 0], mean=396.0153, std=23.1979),
        # 4: right_hip -> left_hip | Table Index (3, 2)
        4: dict(link=("right_hip", "left_hip"), id=4, color=[51, 153, 255], mean=229.9629, std=24.1900),
        # 5: right_wrist -> right_elbow | Table Index (7, 6)
        5: dict(link=("right_wrist", "right_elbow"), id=5, color=[255, 128, 0], mean=263.5240, std=44.5687),
        # 6: right_elbow -> right_shoulder | Table Index (8, 7)
        6: dict(link=("right_elbow", "right_shoulder"), id=6, color=[255, 128, 0], mean=284.6032, std=27.8729),
        # 7: left_shoulder -> left_elbow | Table Index (9, 10)
        7: dict(link=("left_shoulder", "left_elbow"), id=7, color=[0, 255, 0], mean=280.4649, std=18.9359),
        # 8: left_elbow -> left_wrist | Table Index (10, 11)
        8: dict(link=("left_elbow", "left_wrist"), id=8, color=[0, 255, 0], mean=259.1860, std=34.4045),
        # 9: right_hip -> right_shoulder | Table Index (8, 2)
        9: dict(link=("right_hip", "right_shoulder"), id=9, color=[255, 128, 0], mean=577.1766, std=24.3045),
        # 10: left_hip -> left_shoulder | Table Index (9, 3)
        10: dict(link=("left_hip", "left_shoulder"), id=10, color=[0, 255, 0], mean=574.5221, std=22.2362),
        # 11: right_shoulder -> bottom_head | Table Index (12, 8)
        11: dict(link=("right_shoulder", "bottom_head"), id=11, color=[255, 128, 0], mean=223.5728, std=20.4179),
        # 12: left_shoulder -> bottom_head | Table Index (12, 9)
        12: dict(link=("left_shoulder", "bottom_head"), id=12, color=[0, 255, 0], mean=223.0306, std=16.3414),
        # 13: bottom_head -> top_head | Table Index (13, 12)
        13: dict(link=("bottom_head", "top_head"), id=13, color=[51, 153, 255], mean=192.5282, std=33.3035),
    },
}

JOINT_PART_IDS = {
    "panoptic": [
        0,  # 0: neck          -> torso
        1,  # 1: nose          -> head
        0,  # 2: mid_hip       -> torso
        2,  # 3: left_shoulder -> left arm
        2,  # 4: left_elbow    -> left arm
        2,  # 5: left_wrist    -> left arm
        4,  # 6: left_hip      -> left leg
        4,  # 7: left_knee     -> left leg
        4,  # 8: left_ankle    -> left leg
        3,  # 9: right_shoulder -> right arm
        3,  # 10: right_elbow   -> right arm
        3,  # 11: right_wrist   -> right arm
        5,  # 12: right_hip     -> right leg
        5,  # 13: right_knee    -> right leg
        5,  # 14: right_ankle   -> right leg
    ],
}

# Panoptic camera configurations
PANOPTIC_CAM_CONFIGURATIONS = {
    "CMU0": [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)],
    "CMU0(3)": [(0, 3), (0, 6), (0, 12)],
    "CMU0(4)": [(0, 3), (0, 6), (0, 12), (0, 13)],
    "CMU0(6)": [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23), (0, 10)],
    "CMU0(7)": [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23), (0, 10), (0, 16)],
    "CMU1": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 10)],
    "CMU2": [(0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)],
    "CMU3": [(0, 10), (0, 12), (0, 16), (0, 18)],
    "CMU4": [(0, 6), (0, 7), (0, 10), (0, 12)],
}

SHELF_CAM_CONFIGURATIONS = {
    "Shelf5": [0, 1, 2, 3, 4]
}

CAMPUS_CAM_CONFIGURATIONS = {
    "Campus3": [0, 1, 2]
}

MMOR_CAM_CONFIGURATIONS = {
    "MMOR5": [1, 2, 3, 4, 5],
}

def convert_panoptic_to_campus(panoptic_pose: torch.Tensor):
    """
    Transform panoptic order (15 joints) 3D pose to campus dataset order with interpolation.

    :param panoptic_pose: torch.Tensor with shape (*, 15, 3) or (*, 15, 4)
    :return: 3D pose in campus order with shape (*, 14, 3) or (*, 14, 4)
    """
    D = panoptic_pose.shape[-1]
    # Output shape: (*, 14, D)
    output_shape = panoptic_pose.shape[:-2] + (14, D)
    campus_pose = torch.zeros(output_shape, dtype=panoptic_pose.dtype, device=panoptic_pose.device)

    # Mapping for the first 12 joints (Limbs)
    panoptic2campus = torch.tensor([14, 13, 12, 6, 7, 8, 11, 10, 9, 3, 4, 5], device=panoptic_pose.device)
    campus_pose[..., 0:12, :] = panoptic_pose[..., panoptic2campus, :]

    # Head keypoint computation
    neck = panoptic_pose[..., 0, :]  # Panoptic Neck (index 0)
    nose = panoptic_pose[..., 1, :]  # Panoptic Nose (index 1)
    neck_nose_vec = nose[..., :3] - neck[..., :3]  # Neck to nose

    # Estimate head length from skeleton statistics:
    # From SKELETON_INFO["campus"], bottom_head->top_head mean is ~265mm
    # From SKELETON_INFO["panoptic"], nose->neck mean is ~239mm
    head_length_ratio = 265.5409 / 239.0994  # campus head / panoptic neck-nose

    # bottom_head should be slightly above neck (at the base of skull)
    # The neck in Panoptic is at the throat level; bottom_head in Campus is higher
    # Move ~20-30% up from neck toward nose to approximate skull base
    head_bottom = neck + (nose - neck) * 0.3

    # Normalize and scale to expected head length
    vertical_dir = torch.zeros_like(neck_nose_vec)
    vertical_dir[..., 2] = 1.0  # Pure vertical (Z-up)

    # Head top is bottom_head + vertical offset
    # Using the panoptic neck-nose distance scaled by the ratio
    neck_nose_dist = torch.linalg.norm(neck_nose_vec, dim=-1, keepdim=True)
    head_top = head_bottom.clone()
    head_top[..., :3] = head_bottom[..., :3] + vertical_dir * neck_nose_dist * head_length_ratio * 0.8

    # Handle the 4th dimension (confidence/visibility) if present
    if D == 4:
        campus_pose[..., 12, :3] = head_bottom[..., :3]
        campus_pose[..., 13, :3] = head_top[..., :3]
        # Average confidence from neck and nose
        campus_pose[..., 12, 3] = (neck[..., 3] + nose[..., 3]) / 2
        campus_pose[..., 13, 3] = (neck[..., 3] + nose[..., 3]) / 2
    else:
        campus_pose[..., 12, :] = head_bottom
        campus_pose[..., 13, :] = head_top

    return campus_pose


def convert_panoptic_to_shelf(panoptic_pose: torch.Tensor):
    """
    Transform panoptic order (15 joints) 3D pose to shelf dataset order.

    This performs a generic conversion WITHOUT actor-specific head adjustment.
    Use `adjust_shelf_head_for_actor` afterwards to apply actor-specific corrections.

    :param panoptic_pose: torch.Tensor with shape (*, 15, 3) or (*, 15, 4)
    :return: 3D pose in shelf order with shape (*, 14, 3) or (*, 14, 4)
             Also returns intermediate values needed for head adjustment:
             (shelf_pose, head_bottom, dir_face)
    """
    D = panoptic_pose.shape[-1]
    output_shape = panoptic_pose.shape[:-2] + (14, D)
    shelf_pose = torch.zeros(output_shape, dtype=panoptic_pose.dtype, device=panoptic_pose.device)

    # Mapping for the first 12 joints (limbs)
    panoptic2campus = torch.tensor([14, 13, 12, 6, 7, 8, 11, 10, 9, 3, 4, 5], device=panoptic_pose.device)
    shelf_pose[..., 0:12, :] = panoptic_pose[..., panoptic2campus, :]

    # Compute head_bottom (slightly above neck towards nose)
    neck = panoptic_pose[..., 0, :]
    nose = panoptic_pose[..., 1, :]

    head_bottom = neck.clone()
    head_bottom[..., :3] = neck[..., :3] + (nose[..., :3] - neck[..., :3]) * 0.3

    # Compute face direction (needed for actor-specific adjustment later)
    dir_face = nose[..., :3] - head_bottom[..., :3]
    dir_face = dir_face / (torch.linalg.norm(dir_face, dim=-1, keepdim=True) + 1e-8)

    # Store head_bottom in joint 12 (will be kept as-is)
    if D == 4:
        shelf_pose[..., 12, :3] = head_bottom[..., :3]
        shelf_pose[..., 12, 3] = (neck[..., 3] + nose[..., 3]) / 2
        # Joint 13 (head_top) left as zeros - must be filled by adjust_shelf_head_for_actor
        shelf_pose[..., 13, 3] = (neck[..., 3] + nose[..., 3]) / 2
    else:
        shelf_pose[..., 12, :] = head_bottom
        # Joint 13 left as zeros

    return shelf_pose, head_bottom[..., :3], dir_face


def adjust_shelf_head_for_actor(
    shelf_pose: torch.Tensor,
    head_bottom: torch.Tensor,
    dir_face: torch.Tensor,
    actor_id: int,
    head_len: float = 192.0,
) -> torch.Tensor:
    """
    Apply actor-specific head adjustment to a shelf pose.

    The Shelf dataset has different head annotation styles per actor, requiring
    different vertical weight factors for the head direction.

    :param shelf_pose: torch.Tensor with shape (J, 3) or (J, 4) - single pose in shelf format
    :param head_bottom: torch.Tensor with shape (3,) - head bottom position
    :param dir_face: torch.Tensor with shape (3,) - normalized face direction
    :param actor_id: int (0, 1, 2, or 3) - actor index for W_UP lookup
    :param head_len: float - head length in mm (default 192.0)
    :return: shelf_pose with adjusted head_top (joint 13)
    """
    W_UP_LOOKUP = [10.0, 1.5, 0.375, 0.0]
    w_up = W_UP_LOOKUP[actor_id] if actor_id < len(W_UP_LOOKUP) else 0.0

    # Compute head direction with actor-specific vertical weight
    vertical = torch.zeros(3, dtype=dir_face.dtype, device=dir_face.device)
    vertical[2] = 1.0

    dir_head = dir_face + w_up * vertical
    dir_head = dir_head / (torch.linalg.norm(dir_head) + 1e-8)

    # Compute head_top
    head_top = head_bottom + dir_head * head_len

    # Update joint 13
    adjusted_pose = shelf_pose.clone()
    adjusted_pose[13, :3] = head_top

    return adjusted_pose