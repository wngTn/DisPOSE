import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric


class RootMetrics(Metric):
    """
    Center-point evaluation:

    - MDE (mean distance error) + AP@{t} + Recall@{t} for true GT
    - pseudo_MDE + pseudo_AP@{t} + pseudo_Recall@{t} for pseudo-GT
    """

    def __init__(self, root_idx: int | tuple[int, int], dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.root_idx = root_idx

        self.name = "Center Point Metrics"
        self.thresholds = [25, 50, 75, 100, 125, 150]
        self.recall_thresholds = [500]  # Additional recall-only thresholds (in mm)

        # prediction vs ground_truth
        self.add_state("mde", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=[], dist_reduce_fx="cat")
        self.add_state("gt_id", default=[], dist_reduce_fx="cat")
        self.add_state("total_gt", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        # prediction vs pseudo_ground_truth
        self.add_state("pseudo_mde", default=[], dist_reduce_fx="cat")
        self.add_state("pseudo_score", default=[], dist_reduce_fx="cat")
        self.add_state("pseudo_gt_id", default=[], dist_reduce_fx="cat")
        self.add_state("total_pseudo_gt", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        # Metric name lists (for logging / reference)
        self.centerpoint_metrics = (
            ["MDE"]
            + [f"AP@{t}" for t in self.thresholds]
            + [f"Recall@{t}" for t in self.thresholds]
            + [f"Recall@{t}" for t in self.recall_thresholds]
        )
        self.pseudo_metrics = (
            ["pseudo_MDE"]
            + [f"pseudo_AP@{t}" for t in self.thresholds]
            + [f"pseudo_Recall@{t}" for t in self.thresholds]
            + [f"pseudo_Recall@{t}" for t in self.recall_thresholds]
        )
        self.metrics = self.centerpoint_metrics + self.pseudo_metrics

    # -------------------------------------------------------------------------
    # TorchMetrics hooks
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        prediction: Tensor,
        confidence: Tensor,
        global_ids: Tensor,
        pseudo_ground_truth: Tensor,
        ground_truth: Tensor,
    ) -> None:
        """
        Args:
            prediction:         (B, N, 3)       predicted center points.
            confidence:         (B, N)          predicted confidences.
            global_ids:         (B, N)          global GT ids (-1 -> no GT).
            pseudo_ground_truth:(B, N, 4)       (x, y, z, valid > 0).
            ground_truth:       (B, N, J, 4)    (x, y, z, valid > 0).
        """
        assert prediction.dim() == 3 and prediction.size(-1) == 3
        assert confidence.shape[:2] == prediction.shape[:2]
        assert global_ids.shape[:2] == prediction.shape[:2]
        assert pseudo_ground_truth.shape[:2] == prediction.shape[:2] and pseudo_ground_truth.size(-1) == 4
        assert ground_truth.shape[:2] == prediction.shape[:2] and ground_truth.size(-1) == 4

        device = prediction.device

        # Extract root joints
        if isinstance(self.root_idx, int):
            ground_truth = ground_truth[..., self.root_idx, :]  # (B, N, 4)
        else:
            ground_truth = ground_truth[..., list(self.root_idx), :].mean(dim=-2)  # (B, N, 4)

        for pred_pts, pred_scores, gt_ids, pseudo_gt_pts, gt_pts in zip(
            prediction, confidence, global_ids, pseudo_ground_truth, ground_truth
        ):
            # ----------------- prediction vs ground_truth -----------------
            gt_valid_mask = (gt_ids != -1) & (gt_pts[:, 3] > 0)
            if not torch.any(gt_valid_mask):
                continue

            valid_gt_pts = gt_pts[gt_valid_mask, :3]  # (G, 3)
            valid_gt_ids = gt_ids[gt_valid_mask]  # (G,)
            self.total_gt += valid_gt_pts.shape[0]

            pred_valid_mask = pred_scores > 0
            if not torch.any(pred_valid_mask):
                continue

            valid_pred_pts = pred_pts[pred_valid_mask]  # (P, 3)
            valid_pred_scores = pred_scores[pred_valid_mask]  # (P,)

            dists_gt = torch.cdist(valid_pred_pts, valid_gt_pts, p=2)  # (P, G)
            batch_mdes, min_gt_indices = torch.min(dists_gt, dim=1)  # (P,), (P,)
            batch_gt_ids = valid_gt_ids[min_gt_indices]  # (P,)

            self.mde.append(batch_mdes.detach().to(device))
            self.score.append(valid_pred_scores.detach().to(device))
            self.gt_id.append(batch_gt_ids.detach().to(device))

            # ---------------- prediction vs pseudo_ground_truth ----------------
            pseudo_valid_mask = (gt_ids != -1) & (pseudo_gt_pts[:, 3] > 0)
            if not torch.any(pseudo_valid_mask):
                continue

            valid_pseudo_gt_pts = pseudo_gt_pts[pseudo_valid_mask, :3]  # (G', 3)
            valid_pseudo_gt_ids = gt_ids[pseudo_valid_mask]  # (G',)
            self.total_pseudo_gt += valid_pseudo_gt_pts.shape[0]

            dists_pseudo = torch.cdist(valid_pred_pts, valid_pseudo_gt_pts, p=2)  # (P, G')
            batch_pseudo_mdes, min_pseudo_indices = torch.min(dists_pseudo, dim=1)  # (P,), (P,)
            batch_pseudo_gt_ids = valid_pseudo_gt_ids[min_pseudo_indices]  # (P,)

            self.pseudo_mde.append(batch_pseudo_mdes.detach().to(device))
            self.pseudo_score.append(valid_pred_scores.detach().to(device))
            self.pseudo_gt_id.append(batch_pseudo_gt_ids.detach().to(device))

    @torch.no_grad()
    def compute(self) -> dict:
        """
        Returns a dict with:
          - MDE, AP@{t}, Recall@{t}
          - pseudo_MDE, pseudo_AP@{t}, pseudo_Recall@{t}
        """
        results: dict[str, float] = {}
        device = self.total_gt.device

        # prediction vs GT
        mde = self._flatten_state(self.mde, device)
        score = self._flatten_state(self.score, device)
        gt_id = self._flatten_state(self.gt_id, device)
        total_gt = int(self.total_gt.detach())

        gt_metrics = self._compute_block(mde, score, gt_id, total_gt, prefix="")
        results.update(gt_metrics)

        # prediction vs pseudo GT
        pseudo_mde = self._flatten_state(self.pseudo_mde, device)
        pseudo_score = self._flatten_state(self.pseudo_score, device)
        pseudo_gt_id = self._flatten_state(self.pseudo_gt_id, device)
        total_pseudo_gt = int(self.total_pseudo_gt.detach())

        pseudo_metrics = self._compute_block(pseudo_mde, pseudo_score, pseudo_gt_id, total_pseudo_gt, prefix="pseudo_")
        results.update(pseudo_metrics)

        return results

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _flatten_state(state, device: torch.device) -> Tensor:
        """Convert a state (list or tensor) into a flat 1D tensor on `device`."""
        if isinstance(state, list):
            if not state:
                return torch.empty(0, device=device)
            return torch.cat(state, dim=0).to(device)
        if state.numel() == 0:
            return state.to(device)
        return state.view(-1).to(device)

    def _compute_block(
        self,
        mde: Tensor,
        score: Tensor,
        gt_id: Tensor,
        total_gt: int,
        prefix: str = "",
    ) -> dict:
        """
        Shared logic for:
          - (mde, score, gt_id, total_gt, prefix="")
          - (pseudo_mde, pseudo_score, pseudo_gt_id, total_pseudo_gt, prefix="pseudo_")
        """
        total_gt_val = int(total_gt)

        # If nothing accumulated, return default values
        if total_gt_val == 0 or mde.numel() == 0 or score.numel() == 0 or gt_id.numel() == 0:
            base = {f"{prefix}MDE": float("inf"), f"{prefix}MAP": 0.0}
            base.update({f"{prefix}AP@{t}": 0.0 for t in self.thresholds})
            base.update({f"{prefix}Recall@{t}": 0.0 for t in self.thresholds})
            base.update({f"{prefix}Recall@{t}": 0.0 for t in self.recall_thresholds})
            return base

        # Flatten and align lengths
        mde = mde.detach().reshape(-1)
        score = score.detach().reshape(-1)
        gt_id = gt_id.detach().reshape(-1)

        L = min(mde.numel(), score.numel(), gt_id.numel())
        mde = mde[:L]
        score = score[:L]
        gt_id = gt_id[:L]

        # Build eval_list once
        mde_list = mde.cpu().tolist()
        score_list = score.cpu().tolist()
        gt_list = gt_id.cpu().tolist()
        eval_list = [{"mde": mde_list[i], "score": score_list[i], "gt_id": int(gt_list[i])} for i in range(L)]

        # AP and Recall per threshold
        aps = []
        recs = []
        for t in self.thresholds:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt_val, t)
            aps.append(ap)
            recs.append(rec)

        # Additional recall-only thresholds (e.g., Recall@500)
        extra_recs = []
        for t in self.recall_thresholds:
            rec = self._eval_list_to_recall(eval_list, total_gt_val, t)
            extra_recs.append(rec)

        # Overall MDE
        mde_val = self._eval_list_to_mde(eval_list)

        # Build result dict
        results: dict[str, float] = {f"{prefix}MDE": mde_val}
        results[f"{prefix}MAP"] = float(np.mean(aps))
        for t, ap, rec in zip(self.thresholds, aps, recs):
            results[f"{prefix}AP@{t}"] = ap
            results[f"{prefix}Recall@{t}"] = rec
        for t, rec in zip(self.recall_thresholds, extra_recs):
            results[f"{prefix}Recall@{t}"] = rec

        return results

    # -------------------------------------------------------------------------
    # Legacy evaluation routines (MDE-based)
    # -------------------------------------------------------------------------
    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt: int, threshold: float):
        """
        Average Precision (AP) and Average Recall (AR) at a given distance threshold.

        Legacy behavior:
        - detections sorted by descending score
        - a GT can be detected only once:
            * first detection with mde < threshold → TP
            * subsequent detections for the same GT → FP
        """
        if not eval_list or total_gt == 0:
            return 0.0, 0.0

        eval_list.sort(key=lambda k: k["score"], reverse=True)
        num_det = len(eval_list)

        tp = np.zeros(num_det)
        fp = np.zeros(num_det)
        gt_det = set()

        for i, item in enumerate(eval_list):
            if item["mde"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.add(item["gt_id"])
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

        # Legacy AR: recall just before the final appended 1.0
        ar = recall[-2] if recall.size > 2 else 0.0

        return float(ap), float(ar)

    @staticmethod
    def _eval_list_to_mde(eval_list, threshold: float = 500.0) -> float:
        """
        Legacy MDE:

        - sort detections by score (high → low)
        - for each GT, take only the first detection with mde < threshold
        - return mean MDE over those detections
        """
        if not eval_list:
            return float("inf")

        eval_list.sort(key=lambda k: k["score"], reverse=True)

        gt_det = set()
        mde_values = []

        for item in eval_list:
            if item["mde"] < threshold and item["gt_id"] not in gt_det:
                mde_values.append(item["mde"])
                gt_det.add(item["gt_id"])

        return float(np.mean(mde_values)) if mde_values else float("inf")

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt: int, threshold: float = 500.0) -> float:
        """
        Recall at a given MDE threshold.
        Counts unique GT ids that have at least one detection with mde < threshold.
        """
        if total_gt == 0:
            return 0.0

        gt_ids_below_threshold = [e["gt_id"] for e in eval_list if e["mde"] < threshold]
        unique_detected_gt_ids = set(gt_ids_below_threshold)
        return float(len(unique_detected_gt_ids) / total_gt)