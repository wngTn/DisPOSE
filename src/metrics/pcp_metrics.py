# pcp.py

from collections import OrderedDict

import torch
from torch import Tensor
from torchmetrics import Metric

from src.utils.paramUtil import (
    convert_panoptic_to_shelf,
    convert_panoptic_to_campus,
    adjust_shelf_head_for_actor,
)


class PCPMetrics(Metric):
    """
    PCP (Percentage of Correct Parts) Metrics for Multi-person 3D Pose.

    Matches the reference Shelf/Campus evaluation:
    - Iterates over GT actors (person 0, 1, 2, ...)
    - For each GT, finds best matching prediction
    - Accumulates correct/total parts per actor index

    Multi-GPU (DDP) compatible.
    """

    correct_parts: Tensor
    total_parts: Tensor
    bone_correct_parts: Tensor

    total_gt_count: Tensor
    matched_gt_count: Tensor

    def __init__(
        self,
        skeleton_type: str,
        num_actors: int = 3,
        dist_sync_on_step: bool = False,
        recall_threshold: float = 500.0,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "PCP"
        self.skeleton_type = skeleton_type
        self.num_actors = num_actors
        self.recall_threshold = recall_threshold
        self.alpha = 0.5

        self.add_state(
            "correct_parts",
            default=torch.zeros(num_actors, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_parts",
            default=torch.zeros(num_actors, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "bone_correct_parts",
            default=torch.zeros(num_actors, 10, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_gt_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "matched_gt_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

        self.limbs = [
            [0, 1],  # 0: Right Lower leg
            [1, 2],  # 1: Right Upper leg
            [3, 4],  # 2: Left Lower leg
            [4, 5],  # 3: Left Upper leg
            [6, 7],  # 4: Right Upper arm
            [7, 8],  # 5: Right Lower arm
            [9, 10],  # 6: Left Upper arm
            [10, 11],  # 7: Left Lower arm
            [12, 13],  # 8: Head
        ]

        self.bone_groups = OrderedDict(
            [
                ("Head", [8]),
                ("Torso", [9]),
                ("Upper arms", [5, 6]),
                ("Lower arms", [4, 7]),
                ("Upper legs", [1, 2]),
                ("Lower legs", [0, 3]),
            ]
        )

    def _convert_predictions(self, pred_xyz: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Convert predictions from Panoptic to target skeleton format.

        Returns:
            For shelf: (converted_poses, head_bottoms, dir_faces)
            For campus: (converted_poses, None, None)
        """
        if self.skeleton_type == "shelf":
            return convert_panoptic_to_shelf(pred_xyz)
        elif self.skeleton_type == "campus":
            return convert_panoptic_to_campus(pred_xyz), None, None
        else:
            return pred_xyz, None, None

    def _get_adjusted_prediction(
        self,
        pred: Tensor,
        head_bottom: Tensor | None,
        dir_face: Tensor | None,
        actor_id: int,
    ) -> Tensor:
        """
        Get actor-adjusted prediction for PCP computation.

        For shelf skeleton, applies actor-specific head adjustment.
        For other skeletons, returns the prediction unchanged.
        """
        if self.skeleton_type == "shelf" and head_bottom is not None and dir_face is not None:
            return adjust_shelf_head_for_actor(pred, head_bottom, dir_face, actor_id)
        return pred

    @torch.no_grad()
    def update(
        self,
        prediction: Tensor,
        ground_truth: Tensor,
        gt_actor_indices: Tensor,
        **kwargs,
    ) -> None:
        """
        Update metric states.

        Args:
            prediction: (B, N_pred, J, 3) or (B, N_pred, J, 4) predicted 3D keypoints.
                        If 4 channels, last channel is confidence (>= 0 means valid).
            ground_truth: (B, N_gt, J, 3) or (B, N_gt, J, 4) GT keypoints.
                          If 4 channels, last channel is visibility.
            gt_actor_indices: (B, N_gt) actor index (0, 1, 2, ...) for each GT pose.
                              Padded entries should be -1 and will be skipped.
        """
        # Extract confidence from prediction if present
        if prediction.shape[-1] == 4:
            pred_conf = prediction[..., 0, 3]  # (B, N_pred)
            pred_xyz = prediction[..., :3]
        else:
            pred_conf = None
            pred_xyz = prediction

        # Extract xyz from ground truth if visibility channel present
        if ground_truth.shape[-1] == 4:
            gt_xyz = ground_truth[..., :3]
        else:
            gt_xyz = ground_truth

        # Convert predictions if needed (Panoptic 15 -> Shelf/Campus 14)
        needs_conversion = gt_xyz.shape[2] == 14 and pred_xyz.shape[2] == 15
        if needs_conversion:
            pred_xyz, head_bottoms, dir_faces = self._convert_predictions(pred_xyz)
        else:
            head_bottoms, dir_faces = None, None

        # Process each batch element
        for b in range(prediction.shape[0]):
            preds = pred_xyz[b]  # (N_pred, J, 3)
            gts = gt_xyz[b]  # (N_gt, J, 3)
            actor_indices = gt_actor_indices[b]  # (N_gt,)

            # Get head adjustment data for this batch if available
            batch_head_bottoms = head_bottoms[b] if head_bottoms is not None else None
            batch_dir_faces = dir_faces[b] if dir_faces is not None else None

            # Filter valid predictions by confidence
            if pred_conf is not None:
                valid_pred_mask = pred_conf[b] >= 0
                preds = preds[valid_pred_mask]
                if batch_head_bottoms is not None:
                    batch_head_bottoms = batch_head_bottoms[valid_pred_mask]
                    batch_dir_faces = batch_dir_faces[valid_pred_mask]

            # Filter valid GT entries
            valid_gt_mask = (actor_indices >= 0) & (actor_indices < self.num_actors)
            valid_gt_indices = torch.where(valid_gt_mask)[0]

            if valid_gt_indices.shape[0] == 0:
                continue

            # Handle case with no predictions
            if preds.shape[0] == 0:
                for gt_idx in valid_gt_indices:
                    actor_id = actor_indices[gt_idx].item()
                    self.total_parts[actor_id] += 10
                    self.total_gt_count += 1
                continue

            # Match each GT to best prediction and compute PCP
            for gt_idx in valid_gt_indices:
                gt = gts[gt_idx]  # (J, 3)
                actor_id = int(actor_indices[gt_idx].item())

                # Find best matching prediction via MPJPE
                diff = preds - gt.unsqueeze(0)  # (N_pred, J, 3)
                mpjpes = torch.sqrt((diff**2).sum(dim=-1)).mean(dim=-1)  # (N_pred,)
                min_mpjpe, min_idx = torch.min(mpjpes, dim=0)

                # Get matched prediction and apply actor-specific adjustment
                matched_pred = preds[min_idx]  # (J, 3)
                if batch_head_bottoms is not None:
                    matched_head_bottom = batch_head_bottoms[min_idx]
                    matched_dir_face = batch_dir_faces[min_idx]
                    matched_pred = self._get_adjusted_prediction(
                        matched_pred, matched_head_bottom, matched_dir_face, actor_id
                    )

                # Update recall stats
                self.total_gt_count += 1
                if min_mpjpe < self.recall_threshold:
                    self.matched_gt_count += 1

                # Compute PCP for each limb
                self._compute_pcp_for_pose(gt, matched_pred, actor_id)

    def _compute_pcp_for_pose(self, gt: Tensor, pred: Tensor, actor_id: int) -> None:
        """Compute PCP for all limbs of a single pose pair."""
        # Standard limbs
        for limb_idx, (src, dst) in enumerate(self.limbs):
            self.total_parts[actor_id] += 1

            limb_length = torch.norm(gt[src] - gt[dst])
            error_s = torch.norm(pred[src] - gt[src])
            error_e = torch.norm(pred[dst] - gt[dst])

            if (error_s + error_e) / 2.0 <= self.alpha * limb_length:
                self.correct_parts[actor_id] += 1
                self.bone_correct_parts[actor_id, limb_idx] += 1

        # Torso (limb index 9): Hip center -> Head/Neck (joint 12)
        self.total_parts[actor_id] += 1

        pred_hip = (pred[2] + pred[3]) / 2.0
        gt_hip = (gt[2] + gt[3]) / 2.0

        limb_length = torch.norm(gt_hip - gt[12])
        error_s = torch.norm(pred_hip - gt_hip)
        error_e = torch.norm(pred[12] - gt[12])

        if (error_s + error_e) / 2.0 <= self.alpha * limb_length:
            self.correct_parts[actor_id] += 1
            self.bone_correct_parts[actor_id, 9] += 1

    @torch.no_grad()
    def compute(self) -> dict:
        """
        Compute final PCP metrics.

        Returns:
            Dictionary containing:
            - Actor 1, 2, 3 PCP (and more if num_actors > 3)
            - Average PCP (mean of first 3 actors, matching reference)
            - Per-bone-group PCP
            - Recall@Threshold
        """
        results = {}

        actor_pcp = self.correct_parts / (self.total_parts + 1e-8)

        for i in range(self.num_actors):
            results[f"Actor {i + 1}"] = actor_pcp[i].item()

        results["Average"] = actor_pcp[:3].mean().item()

        parts_per_bone = self.total_parts / 10
        for group_name, bone_indices in self.bone_groups.items():
            group_correct = self.bone_correct_parts[:, bone_indices].sum(dim=-1)
            group_total = parts_per_bone * len(bone_indices)
            group_pcp = group_correct / (group_total + 1e-8)
            results[f"Bone/{group_name}"] = group_pcp[:3].mean().item()

        total_gt = self.total_gt_count.item()
        matched_gt = self.matched_gt_count.item()
        results[f"Recall@{int(self.recall_threshold)}"] = matched_gt / (total_gt + 1e-8)

        return results
