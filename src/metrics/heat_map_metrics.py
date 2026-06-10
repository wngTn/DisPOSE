import torch
from torch import Tensor
from torchmetrics import Metric

from src.losses.custom.pseudo_regression_loss import compute_heatmap_vectorized


class HeatMapL2(Metric):
    def __init__(
        self,
        sigma: float = 2.0,
        image_size: tuple[int, int] = (960, 512),
        heatmap_size: tuple[int, int] = (240, 128),
        dist_sync_on_step: bool = True,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = "HeatMapL2"
        self.sigma = sigma
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("MSE", default=torch.zeros(1), dist_reduce_fx="sum")
        self.metrics = ["MSE"]

    def compute(self):
        count = self.count
        heat_map_metrics = {
            "MSE": self.MSE / count.clamp(min=1),
        }
        return heat_map_metrics

    @staticmethod
    def _apply_affine(points: Tensor, affine: Tensor) -> Tensor:
        """
        Apply affine transformation to points.

        Args:
            points: (*, V, N, J, 2)
            affine: (*, V, 2, 3)

        Returns:
            Transformed points: (*, V, N, J, 2)
        """
        ones = torch.ones_like(points[..., :1])
        points_h = torch.cat([points, ones], dim=-1)  # (*, V, N, J, 3)
        affine_exp = affine.unsqueeze(-3).unsqueeze(-3)  # (*, V, 1, 1, 2, 3)
        points_h = points_h.unsqueeze(-1)  # (*, V, N, J, 3, 1)
        transformed = torch.matmul(affine_exp, points_h).squeeze(-1)  # (*, V, N, J, 2)
        return transformed

    def update(
        self,
        prediction: Tensor,  # (*, V, J, H, W)
        gt_keypoints_xys: Tensor,  # (*, V, N, J, 3)
        affine_transforms: Tensor,  # (*, V, 2, 3)
    ):
        # Check if gt keypoint joints have the same format as the prediction
        # prediction.shape[-3] is J (joints), gt_keypoints_xys.shape[-2] is J
        if prediction.shape[-3] != gt_keypoints_xys.shape[-2]:
            # Convert ground truth shelf/campus -> panoptic format
            gt_keypoints_xys = convert_campus_to_panoptic_2d(gt_keypoints_xys)

        gt_xy = gt_keypoints_xys[..., :2]  # (*, V, N, J, 2)
        gt_score = gt_keypoints_xys[..., 2]  # (*, V, N, J)

        valid_mask = gt_score > 0

        if not valid_mask.any():
            return

        # Apply affine transform to GT keypoints
        gt_xy_transformed = self._apply_affine(gt_xy, affine_transforms)

        # Generate target heatmaps (matching loss behavior)
        target_heatmaps = compute_heatmap_vectorized(
            joints=gt_xy_transformed,
            joints_vis=valid_mask,
            sigma=self.sigma,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            per_person=False,  # Output: (*, V, J, H, W)
            joint_weights=gt_score.clamp(0, 1),  # Match loss behavior
        )

        # Count joints with at least one visible person
        joint_has_supervision = valid_mask.any(dim=-2)  # (*, V, J)
        num_supervised = joint_has_supervision.sum()

        if num_supervised <= 0:
            return

        # Compute MSE only on supervised joints
        diff_sq = (prediction - target_heatmaps) ** 2  # (*, V, J, H, W)

        # Mask to only count supervised joints
        mask = joint_has_supervision.unsqueeze(-1).unsqueeze(-1)  # (*, V, J, 1, 1)
        masked_diff_sq = diff_sq * mask

        H, W = prediction.shape[-2:]
        mse = masked_diff_sq.sum() / (num_supervised * H * W)

        self.count += 1
        self.MSE += mse.detach().cpu()

def convert_campus_to_panoptic_2d(campus_pose: torch.Tensor) -> torch.Tensor:
    """
    Transform Campus/Shelf order (14 joints) to Panoptic order (15 joints) for 2D poses.
    
    Args:
        campus_pose: torch.Tensor with shape (*, 14, 2) or (*, 14, 3) where last dim is (x, y) or (x, y, score)
        
    Returns:
        Panoptic pose with shape (*, 15, 2) or (*, 15, 3)
        
    Note:
        - mid_hip (joint 2) is computed as average of left_hip and right_hip
        - neck (joint 0) is estimated as shoulder midpoint
        - nose (joint 1) is estimated from neck and head_bottom
    """
    D = campus_pose.shape[-1]  # 2 or 3
    output_shape = campus_pose.shape[:-2] + (15, D)
    panoptic_pose = torch.zeros(output_shape, dtype=campus_pose.dtype, device=campus_pose.device)
    
    # Limb joints (direct mapping)
    panoptic_pose[..., 14, :] = campus_pose[..., 0, :]   # right_ankle
    panoptic_pose[..., 13, :] = campus_pose[..., 1, :]   # right_knee
    panoptic_pose[..., 12, :] = campus_pose[..., 2, :]   # right_hip
    panoptic_pose[..., 6, :] = campus_pose[..., 3, :]    # left_hip
    panoptic_pose[..., 7, :] = campus_pose[..., 4, :]    # left_knee
    panoptic_pose[..., 8, :] = campus_pose[..., 5, :]    # left_ankle
    panoptic_pose[..., 11, :] = campus_pose[..., 6, :]   # right_wrist
    panoptic_pose[..., 10, :] = campus_pose[..., 7, :]   # right_elbow
    panoptic_pose[..., 9, :] = campus_pose[..., 8, :]    # right_shoulder
    panoptic_pose[..., 3, :] = campus_pose[..., 9, :]    # left_shoulder
    panoptic_pose[..., 4, :] = campus_pose[..., 10, :]   # left_elbow
    panoptic_pose[..., 5, :] = campus_pose[..., 11, :]   # left_wrist
    
    # Extract relevant joints
    left_hip = campus_pose[..., 3, :]
    right_hip = campus_pose[..., 2, :]
    left_shoulder = campus_pose[..., 9, :]
    right_shoulder = campus_pose[..., 8, :]
    head_bottom = campus_pose[..., 12, :]
    
    # mid_hip: average of left and right hip
    mid_hip_xy = (left_hip[..., :2] + right_hip[..., :2]) / 2.0
    panoptic_pose[..., 2, :2] = mid_hip_xy
    if D == 3:
        panoptic_pose[..., 2, 2] = torch.minimum(left_hip[..., 2], right_hip[..., 2])
    
    # neck: shoulder midpoint
    neck_xy = (left_shoulder[..., :2] + right_shoulder[..., :2]) / 2.0
    panoptic_pose[..., 0, :2] = neck_xy
    if D == 3:
        panoptic_pose[..., 0, 2] = torch.minimum(left_shoulder[..., 2], right_shoulder[..., 2])
    
    # nose: estimated from neck and head_bottom
    # From forward: head_bottom = neck + (nose - neck) * 0.3
    # Therefore: nose = neck + (head_bottom - neck) / 0.3
    nose_xy = neck_xy + (head_bottom[..., :2] - neck_xy) / 0.3
    panoptic_pose[..., 1, :2] = nose_xy
    if D == 3:
        panoptic_pose[..., 1, 2] = head_bottom[..., 2]
    
    return panoptic_pose