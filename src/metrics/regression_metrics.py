import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from src.utils.paramUtil import convert_panoptic_to_campus


class RegressionMetrics(Metric):
    """
    Multi-person 3D regression metrics:

    - MPJPE (legacy-style, one detection per GT)
    - AP@{threshold} and Recall@{threshold} for multiple MPJPE thresholds
    """

    mpjpe: list[Tensor]
    score: list[Tensor]
    gt_id: list[Tensor]
    total_gt: Tensor

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "Regression"
        # MPJPE thresholds in millimeters
        self.thresholds = [25, 50, 75, 100, 125, 150]
        self.recall_only_thresholds = [500]

        # Variable-length states → store as lists of 1D tensors (flatten in compute).
        self.add_state("mpjpe", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=[], dist_reduce_fx="cat")
        self.add_state("gt_id", default=[], dist_reduce_fx="cat")

        # Total number of GT poses (scalar, summed across processes).
        self.add_state("total_gt", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        # Metric names (for reference / logging)
        self.keypoint_metrics = (
            ["MPJPE", "MAP", "MRecall"]
            + [f"AP@{t}" for t in self.thresholds]
            + [f"Recall@{t}" for t in self.thresholds]
            + [f"Recall@{t}" for t in self.recall_only_thresholds]
        )
        self.metrics = self.keypoint_metrics

    # -------------------------------------------------------------------------
    # TorchMetrics hooks
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        prediction: Tensor,
        confidence: Tensor,
        global_ids: Tensor,
        ground_truth: Tensor,
        **kwargs,
    ) -> None:
        """
        Update metric states with a new batch.

        Args:
            prediction:  (B, N, J, 3) predicted 3D keypoints (x, y, z).
            confidence:  (B, N) predicted pose confidences.
            global_ids:  (B, N) global GT IDs for each pose, -1 means "no GT".
            ground_truth: (B, N, J, 4) GT keypoints (x, y, z, visibility).
        """
        assert prediction.dim() == 4, "prediction must have shape (B, N, J, 3)"
        assert confidence.dim() == 2, "confidence must have shape (B, N)"
        assert ground_truth.dim() == 4, "ground_truth must have shape (B, N, J, 4)"

        B, N, J, _ = prediction.shape
        assert confidence.shape == (B, N), "confidence shape must match (B, N)"
        assert global_ids.shape == (B, N), "global_ids shape must be (B, N)"

        device = prediction.device

        # Convert prediction from panoptic (15 joints) to shelf/campus (14 joints).
        # Use convert_panoptic_to_campus for BOTH shelf and campus: joints 0-12
        # are computed identically across the two converters, but the campus one
        # also fills joint 13 (top_head) via a vertical-offset heuristic instead
        # of leaving it at zero. The shelf converter's `adjust_shelf_head_for_actor`
        # would be more accurate but requires actor identity that isn't available
        # at metric time; leaving j13 at zero (the previous behavior) was injecting
        # a permanent ~1700 mm error per pose into MPJPE and dragging AP to 0.
        if ground_truth.shape[2] == 14 and prediction.shape[2] == 15:
            prediction = convert_panoptic_to_campus(prediction)

        # Process each sample independently
        for preds_xyz, preds_s, target_ids, target_xyzs in zip(prediction, confidence, global_ids, ground_truth):
            # target_xyzs: (N, J, 4), target_ids: (N,)

            # Keep only GT poses with valid IDs
            valid_targets = target_ids != -1
            if not torch.any(valid_targets):
                continue

            gt_kps = target_xyzs[valid_targets]  # (G, J, 4)
            gt_ids = target_ids[valid_targets]  # (G,)

            # Count valid GT poses in this sample
            self.total_gt += valid_targets.sum().to(self.total_gt.device)

            sample_mpjpes = []
            sample_scores = []
            sample_gt_ids = []

            # Loop over predicted poses for this sample
            for pred_pose_xyzs, pred_pose_score in zip(preds_xyz, preds_s):
                # Skip invalid / zero-confidence predictions
                if pred_pose_score <= 0:
                    continue

                # Compute MPJPE to each valid GT pose
                mpjpe_list = []
                for gt_pose in gt_kps:
                    # vis mask: visibility channel > 0.0
                    vis = gt_pose[:, -1] > 0.0
                    if not torch.any(vis):
                        continue

                    diff = pred_pose_xyzs[vis] - gt_pose[vis, :3]
                    mpjpe = torch.mean(torch.linalg.norm(diff, dim=-1))
                    mpjpe_list.append(mpjpe)

                if not mpjpe_list:
                    continue

                mpjpes = torch.stack(mpjpe_list, dim=0)  # (G_valid,)
                min_gt_index = torch.argmin(mpjpes)
                min_mpjpe = mpjpes[min_gt_index]
                matched_gt_id = gt_ids[min_gt_index]

                sample_mpjpes.append(min_mpjpe)
                sample_scores.append(pred_pose_score)
                sample_gt_ids.append(matched_gt_id)

            if sample_mpjpes:
                self.mpjpe.append(torch.stack(sample_mpjpes).detach().to(device))
                self.score.append(torch.stack(sample_scores).detach().to(device))
                self.gt_id.append(torch.stack(sample_gt_ids).detach().to(device))

    @torch.no_grad()
    def compute(self) -> dict:
        """
        Compute final metrics from accumulated states.

        Returns:
            dict with keys:
              - "MPJPE"
              - "AP@{t}" and "Recall@{t}" for each threshold t in self.thresholds
        """
        device = self.total_gt.device
        total_gt = int(self.total_gt.item())

        mpjpe_t = self._flatten_state(self.mpjpe, device)
        score_t = self._flatten_state(self.score, device)
        gt_id_t = self._flatten_state(self.gt_id, device)

        # If nothing accumulated or no GT, return defaults
        if total_gt == 0 or mpjpe_t.numel() == 0 or score_t.numel() == 0 or gt_id_t.numel() == 0:
            return {
                "MPJPE": float("inf"),
                "MAP": 0.0,
                "MRecall": 0.0,
                **{f"AP@{t}": 0.0 for t in self.thresholds},
                **{f"Recall@{t}": 0.0 for t in self.thresholds},
                **{f"Recall@{t}": 0.0 for t in self.recall_only_thresholds},
            }

        # Ensure arrays have the same length
        L = min(mpjpe_t.numel(), score_t.numel(), gt_id_t.numel())
        mpjpe_t = mpjpe_t[:L]
        score_t = score_t[:L]
        gt_id_t = gt_id_t[:L]

        # Build eval_list for downstream AP / MPJPE functions
        eval_list = [
            {
                "mpjpe": float(mpjpe_t[i].item()),
                "score": float(score_t[i].item()),
                "gt_id": int(gt_id_t[i].item()),
            }
            for i in range(L)
        ]

        # AP and Recall per threshold (legacy behavior)
        aps = []
        recs = []
        for t in self.thresholds:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)


        # Overall MPJPE (legacy behavior)
        mpjpe_val = self._eval_list_to_mpjpe(eval_list)

        # Build results dict
        results = {"MPJPE": mpjpe_val}
        results["MAP"] = float(np.mean(aps))
        results["MRecall"] = float(np.mean(recs))
        for t, ap, rec in zip(self.thresholds, aps, recs):
            results[f"AP@{t}"] = ap
            results[f"Recall@{t}"] = rec

        for t in self.recall_only_thresholds:
            _, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            results[f"Recall@{t}"] = rec

        return results

    # -------------------------------------------------------------------------
    # Helper / legacy evaluation routines
    # -------------------------------------------------------------------------
    @staticmethod
    def _flatten_state(state, device: torch.device) -> Tensor:
        """Flatten a metric state that may be a list of tensors or a tensor."""
        if isinstance(state, list):
            if not state:
                return torch.empty(0, device=device)
            return torch.cat(state, dim=0).to(device)
        if state.numel() == 0:
            return state.to(device)
        return state.view(-1).to(device)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt: int, threshold: float):
        """
        Average Precision (AP) and Average Recall (AR) at a given MPJPE threshold.

        - detections sorted by descending score
        - a GT can be detected only once:
            * first detection with mpjpe < threshold → TP
            * subsequent detections for the same GT → FP
        """
        if not eval_list or total_gt == 0:
            return 0.0, 0.0

        # Sort detections by score (high → low)
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        num_det = len(eval_list)

        tp = np.zeros(num_det)
        fp = np.zeros(num_det)
        gt_det = []

        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / (total_gt + 1e-5)
        precision = tp / (tp + fp + 1e-5)

        # Monotonic precision envelope
        for n in range(num_det - 2, -1, -1):
            precision[n] = max(precision[n], precision[n + 1])

        precision = np.concatenate(([0.0], precision, [0.0]))
        recall = np.concatenate(([0.0], recall, [1.0]))

        idx = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])

        # Legacy AR definition: recall just before the final appended 1.0
        ar = recall[-2] if recall.size > 2 else 0.0

        return float(ap), float(ar)

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold: float = 500.0) -> float:
        """
        MPJPE:

        - sort detections by score (high → low)
        - for each GT, take only the first detection with mpjpe < threshold
        - return mean MPJPE over those detections
        """
        if not eval_list:
            return np.inf

        eval_list.sort(key=lambda k: k["score"], reverse=True)

        gt_det = []
        mpjpes = []

        for item in eval_list:
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return float(np.mean(mpjpes)) if mpjpes else np.inf
