import torch
from torch import Tensor
from torchmetrics import Metric


class HeatMapRegressionMetrics(Metric):
    """
    Evaluates:
      - Recall@t: fraction of GT points that have at least one prediction within t pixels
      - AP@t: average precision using prediction scores and 1-to-1 matching within t pixels

    Input format:
      prediction:   (..., 3) = (x, y, score), valid if score > 0
      ground_truth: (..., 3) = (x, y, valid), valid if valid > 0
    """

    def __init__(self, dist_sync_on_step: bool = False, thresholds=(50, 100, 200)):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "Recall + AP Metrics"
        self.thresholds = [float(t) for t in thresholds]

        # ---- Recall state ----
        self.add_state("min_dist_to_gt", default=[], dist_reduce_fx="cat")
        self.add_state("total_gt", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        # ---- AP state (TorchMetrics-compatible) ----
        # We store per-update tensors and later concat.
        # For each threshold we append tp/fp flags and scores. We encode threshold index in a parallel tensor.
        self.add_state("ap_scores", default=[], dist_reduce_fx="cat")  # (K,)
        self.add_state("ap_tp", default=[], dist_reduce_fx="cat")  # (K,)
        self.add_state("ap_fp", default=[], dist_reduce_fx="cat")  # (K,)
        self.add_state("ap_tid", default=[], dist_reduce_fx="cat")  # (K,) long threshold id

        # total GT per threshold id (vector)
        self.add_state(
            "ap_total_gt",
            default=torch.zeros(len(self.thresholds), dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @property
    def metrics(self) -> list[str]:
        return [f"Recall@{int(t)}" for t in self.thresholds] + [f"AP@{int(t)}" for t in self.thresholds]

    @staticmethod
    def _flatten_with_batch(x: Tensor) -> Tensor:
        """
        Ensure (B, N, 3). If input has no batch dim, treat as B=1.
        """
        if x.numel() == 0:
            return x.reshape(1, 0, 3)
        if x.ndim == 2 and x.shape[-1] == 3:
            return x.unsqueeze(0)  # (1, N, 3)
        return x.reshape(x.shape[0], -1, 3)  # assume first dim is batch

    @staticmethod
    def _ap_from_sorted_tp_fp(tp_sorted: Tensor, fp_sorted: Tensor, num_gt: int) -> float:
        """
        AP as area under precision-recall curve using precision envelope.
        """
        if num_gt <= 0:
            return 0.0
        if tp_sorted.numel() == 0:
            return 0.0

        tp_cum = torch.cumsum(tp_sorted, dim=0)
        fp_cum = torch.cumsum(fp_sorted, dim=0)

        recall = tp_cum / float(num_gt)
        precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0)

        # precision envelope
        mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
        mpre = torch.cat(
            [torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)]
        )
        for i in range(mpre.numel() - 2, -1, -1):
            mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

        idx = torch.where(mrec[1:] != mrec[:-1])[0]
        ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
        return float(ap)

    @torch.no_grad()
    def update(self, prediction: Tensor, ground_truth: Tensor) -> None:
        pred = self._flatten_with_batch(prediction)
        gt = self._flatten_with_batch(ground_truth)

        device = gt.device
        B = pred.shape[0]
        T = len(self.thresholds)

        for b in range(B):
            pred_b = pred[b]  # (N, 3)
            gt_b = gt[b]  # (N, 3)

            pred_valid = pred_b[:, 2] > 0
            gt_valid = gt_b[:, 2] > 0

            preds_xy = pred_b[pred_valid, :2]  # (P, 2)
            preds_s = pred_b[pred_valid, 2]  # (P,)
            gts_xy = gt_b[gt_valid, :2]  # (G, 2)

            G = gts_xy.shape[0]
            if G == 0:
                continue

            # ---- Recall bookkeeping ----
            self.total_gt += G
            if preds_xy.shape[0] == 0:
                self.min_dist_to_gt.append(torch.full((G,), float("inf"), device=device))
            else:
                d = torch.cdist(gts_xy, preds_xy, p=2)  # (G, P)
                min_d, _ = torch.min(d, dim=1)
                self.min_dist_to_gt.append(min_d)

            # ---- AP bookkeeping ----
            # count GT for each threshold (same G)
            self.ap_total_gt += torch.tensor([G] * T, device=self.ap_total_gt.device, dtype=torch.long)

            P = preds_xy.shape[0]
            if P == 0:
                continue

            # sort predictions by score (desc)
            order = torch.argsort(preds_s, descending=True)
            preds_xy_s = preds_xy[order]
            preds_s_s = preds_s[order]

            # pairwise distances once: (P, G)
            dists = torch.cdist(preds_xy_s, gts_xy, p=2)

            # For each threshold, compute greedy TP/FP flags (1-to-1 match)
            for tid, thr in enumerate(self.thresholds):
                matched_gt = torch.zeros((G,), dtype=torch.bool, device=device)
                tp_flags = torch.zeros((P,), dtype=torch.float32, device=device)
                fp_flags = torch.zeros((P,), dtype=torch.float32, device=device)

                for i in range(P):
                    within = dists[i] <= thr
                    if not torch.any(within):
                        fp_flags[i] = 1.0
                        continue

                    cand = torch.where(within)[0]
                    cand = cand[~matched_gt[cand]]
                    if cand.numel() == 0:
                        fp_flags[i] = 1.0
                        continue

                    # nearest unmatched GT among candidates
                    j = cand[torch.argmin(dists[i, cand])]
                    matched_gt[j] = True
                    tp_flags[i] = 1.0

                # append to global lists with threshold id
                self.ap_scores.append(preds_s_s.detach())
                self.ap_tp.append(tp_flags.detach())
                self.ap_fp.append(fp_flags.detach())
                self.ap_tid.append(torch.full((P,), tid, device=device, dtype=torch.long))

    @staticmethod
    def _flatten_state(state, device) -> Tensor:
        """torchmetrics dist_reduce_fx='cat' produces a list pre-sync and a tensor post-sync."""
        if isinstance(state, Tensor):
            return state
        if not state:
            return torch.empty(0, device=device)
        return torch.cat(state, dim=0)

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        device = self.total_gt.device

        # ---- Recall ----
        total_gt = int(self.total_gt.item())
        if total_gt == 0:
            for t in self.thresholds:
                results[f"Recall@{int(t)}"] = 0.0
        else:
            min_dists = self._flatten_state(self.min_dist_to_gt, device)
            for t in self.thresholds:
                detected = (min_dists < t).sum().item()
                results[f"Recall@{int(t)}"] = float(detected) / float(total_gt)

        # ---- AP ----
        scores = self._flatten_state(self.ap_scores, device)
        tp = self._flatten_state(self.ap_tp, device)
        fp = self._flatten_state(self.ap_fp, device)
        tid = self._flatten_state(self.ap_tid, device)
        if scores.numel() == 0 or tid.numel() == 0:
            for t in self.thresholds:
                results[f"AP@{int(t)}"] = 0.0
        else:

            # compute AP separately per threshold id
            for i, t in enumerate(self.thresholds):
                num_gt = int(self.ap_total_gt[i].item())
                mask = tid == i
                if num_gt == 0 or not torch.any(mask):
                    results[f"AP@{int(t)}"] = 0.0
                    continue

                s_i = scores[mask]
                tp_i = tp[mask]
                fp_i = fp[mask]

                order = torch.argsort(s_i, descending=True)
                ap = self._ap_from_sorted_tp_fp(tp_i[order], fp_i[order], num_gt)
                results[f"AP@{int(t)}"] = ap

        # convenience aggregates
        results["mAP"] = float(sum(results[f"AP@{int(t)}"] for t in self.thresholds) / len(self.thresholds))
        results["mRecall"] = float(sum(results[f"Recall@{int(t)}"] for t in self.thresholds) / len(self.thresholds))
        return results
