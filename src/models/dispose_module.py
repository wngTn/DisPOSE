import logging
import time
from functools import partial
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf.dictconfig import DictConfig

from src.models.shared.triangulation import triangulate_algebraic
from src.utils.common import is_tensor, move_to_device
from src.utils.resume import load_pretrained_component

from src.metrics import BaseMetrics

log = logging.getLogger(__name__)

# Train-time Gaussian jitter (mm) on teacher-forced reference points.
PSEUDO_REF_JITTER_MM = 64.0


def _format_eta(seconds: float) -> str:
    """Compact `1d23h17m51s` (drop any leading-zero unit, keep seconds)."""
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if days or hours:
        parts.append(f"{hours}h")
    if days or hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return "".join(parts)


def _format_metrics_table(d: dict) -> str:
    """Two-column box-drawn table of metric -> 4-decimal value, sorted by key."""
    items = sorted(((k, f"{float(v):.4f}") for k, v in d.items()), key=lambda kv: kv[0])
    if not items:
        return ""
    key_w = max(max(len(k) for k, _ in items), len("Metric"))
    val_w = max(max(len(v) for _, v in items), len("Value"))
    top = f"┌{'─' * (key_w + 2)}┬{'─' * (val_w + 2)}┐"
    sep = f"├{'─' * (key_w + 2)}┼{'─' * (val_w + 2)}┤"
    bot = f"└{'─' * (key_w + 2)}┴{'─' * (val_w + 2)}┘"
    header = f"│ {'Metric'.ljust(key_w)} │ {'Value'.rjust(val_w)} │"
    rows = [f"│ {k.ljust(key_w)} │ {v.rjust(val_w)} │" for k, v in items]
    return "\n".join([top, header, sep, *rows, bot])


class DisPOSE(L.LightningModule):
    """Multi-person 3D pose estimation LightningModule.

    Composes three sub-networks (all plain nn.Module):
      - backbone: ResNet keypoint/feature extractor
      - root_regression_net: cross-view person ID matching (optional)
      - pose_regression_net: 3D pose regression (optional)
    """

    def __init__(
        self,
        backbone: nn.Module,
        optimizer: partial,
        scheduler: partial,
        metric: DictConfig,
        losses: nn.Module,
        assignment: nn.Module | tuple | None = None,
        regressor: nn.Module | tuple | None = None,
        logs_root: str = "./logs",
        warmup_steps: int = 0,
        camera_setup: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Sub-networks (pure nn.Module)
        self.backbone = backbone
        self._camera_setup = camera_setup

        if assignment is not None:
            self.root_regression_net = self._init_component(assignment, "assignment", logs_root)

        if regressor is not None:
            self.pose_regression_net = self._init_component(regressor, "regressor", logs_root)

        # Losses and metrics
        self._losses = losses
        self._metrics = BaseMetrics(metric)

        # Optimizer / scheduler configs (partials, called in configure_optimizers)
        self._optimizer_cfg = optimizer
        self._scheduler_cfg = scheduler
        self._warmup_steps = warmup_steps

        # Component freezing — three orthogonal knobs:
        #   freeze:                  whole component is frozen (requires_grad=False + always eval)
        #   freeze_bn_running_stats: component is trainable but pinned to eval mode so BN
        #                            running_mean/var don't drift (BN gamma/beta still get gradients)
        #   param_group_lr:          per-component LR override (used by configure_optimizers)
        # The first two are pinned in train() so super().train() can't toggle them back on.
        self._freeze_list: list[str] = list(kwargs.get("freeze") or [])
        self._bn_eval_list: list[str] = list(kwargs.get("freeze_bn_running_stats") or [])
        self._param_group_lr: dict[str, float] = dict(kwargs.get("param_group_lr") or {})

        for component in self._freeze_list:
            net = self._get_component(component)
            if net is not None:
                net.requires_grad_(False)
                net.eval()
        for component in self._bn_eval_list:
            net = self._get_component(component)
            if net is not None:
                net.eval()

        total = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        log.info(f"Parameters: {total:.2f}M total, {trainable:.2f}M trainable")
        if self._bn_eval_list:
            log.info(f"BN-eval pinned (trainable params, frozen running stats): {self._bn_eval_list}")
        if self._param_group_lr:
            log.info(f"Per-component LR overrides: {self._param_group_lr}")

    def _get_component(self, name: str) -> nn.Module | None:
        """Resolve a component name (backbone / assignment / regressor) to its module."""
        if name == "backbone":
            return self.backbone
        if name == "assignment" and hasattr(self, "root_regression_net"):
            return self.root_regression_net
        if name == "regressor" and hasattr(self, "pose_regression_net"):
            return self.pose_regression_net
        return None

    def train(self, mode: bool = True) -> "DisPOSE":
        """Pin frozen and bn-eval components to eval() so super().train() can't toggle them on."""
        super().train(mode)
        if mode:
            for name in (*self._freeze_list, *self._bn_eval_list):
                net = self._get_component(name)
                if net is not None:
                    net.eval()
        return self

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    def _init_component(
        self,
        component_cfg: nn.Module | tuple,
        component_type: str,
        logs_root: str,
    ) -> nn.Module:
        if isinstance(component_cfg, nn.Module):
            return component_cfg
        elif callable(component_cfg):
            return component_cfg()
        else:
            log.info(
                f"Loading pretrained {component_type} from "
                f"'{component_cfg[0]}' at iteration {component_cfg[1]}"
            )
            return load_pretrained_component(
                experiment_spec=component_cfg,
                component_type=component_type,
                logs_root=logs_root,
                camera_setup=self._camera_setup,
            )

    # ------------------------------------------------------------------
    # Lightning: configure_optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Build param groups. If `param_group_lr` overrides exist, the named components
        # get their own group with the specified LR; everything else lands in the default
        # group at the optimizer-config LR. Cosine / warmup schedulers handle multi-group
        # LRs correctly — each group is scaled from its own initial LR.
        if self._param_group_lr:
            param_groups: list[dict] = []
            assigned_param_ids: set[int] = set()
            for name, lr in self._param_group_lr.items():
                net = self._get_component(name)
                if net is None:
                    log.warning(f"param_group_lr: unknown component '{name}', skipped")
                    continue
                params = [p for p in net.parameters() if p.requires_grad]
                if not params:
                    continue
                param_groups.append({"params": params, "lr": float(lr), "name": name})
                assigned_param_ids.update(id(p) for p in params)
            default_params = [
                p for p in self.parameters()
                if p.requires_grad and id(p) not in assigned_param_ids
            ]
            if default_params:
                # Default group inherits LR from the optimizer partial config.
                param_groups.insert(0, {"params": default_params, "name": "default"})
            optimizer = self._optimizer_cfg(params=param_groups)
        else:
            optimizer = self._optimizer_cfg(
                params=filter(lambda p: p.requires_grad, self.parameters())
            )
        main_scheduler = self._scheduler_cfg(optimizer=optimizer)

        if self._warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self._warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, main_scheduler], milestones=[self._warmup_steps]
            )
        else:
            scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ------------------------------------------------------------------
    # Lightning: training_step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        ipts = self._preprocess(batch)
        retval = self._forward_train(ipts)
        loss = self._losses.update(retval)

        # Degenerate batches (e.g. empty hypergraph) produce a zero-loss tensor
        # that — in the worst case — has no autograd graph at all. Under DDP
        # this desynchronises ranks: with one rank skipping every gradient hook
        # while the other issues all-reduce on populated buckets, NCCL collectives
        # mismatch and the watchdog times out 30 min later (rank 0 ALLREDUCE,
        # rank 1 BROADCAST). Anchor the loss to every trainable parameter with a
        # zero coefficient so every DDP grad hook always fires.
        if loss <= 1e-12:
            log.warning("Loss <= 1e-12 (degenerate batch); anchoring zero gradient to all params")
            anchor = sum(p.sum() for p in self.parameters() if p.requires_grad)
            loss = loss + 0.0 * anchor

        # Periodically compute & log averaged losses, then reset accumulator
        log_every = getattr(self.trainer, "log_every_n_steps", 50)
        if (self.global_step + 1) % log_every == 0:
            loss_dict = self._losses.compute()
            self.log_dict(
                {f"train/{k}": v for k, v in loss_dict.items()},
                prog_bar=True, on_step=True, on_epoch=False,
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, prog_bar=False, on_step=True, on_epoch=False)
            if self.trainer.is_global_zero:
                now = time.monotonic()
                last = getattr(self, "_last_log_time", None)
                self._last_log_time = now
                eta_part = ""
                if last is not None and self.trainer.max_steps:
                    steps_per_sec = log_every / max(now - last, 1e-9)
                    remaining = self.trainer.max_steps - (self.global_step + 1)
                    eta_part = f" | eta: {_format_eta(remaining / max(steps_per_sec, 1e-9))}"
                log.info(
                    f"[Step {self.global_step + 1}] Train — "
                    f"loss: {loss.item():.4f} | lr: {lr:.2e}{eta_part}"
                )

        return loss

    # ------------------------------------------------------------------
    # Lightning: validation / test / predict
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        ipts = self._preprocess(batch)
        result_set = self._forward_eval(ipts)
        self._extract_and_update_metrics(result_set, ipts)
        return result_set

    def on_validation_epoch_end(self) -> None:
        metrics_dict = self.compute_metrics()
        if metrics_dict:
            self.log_dict(
                {f"val/{k}": v for k, v in metrics_dict.items()},
                prog_bar=True, sync_dist=True,
            )
            if self.trainer.is_global_zero:
                log.info(f"[Step {self.global_step}] Validation\n{_format_metrics_table(metrics_dict)}")
        self.reset_metrics()

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        ipts = self._preprocess(batch)
        result_set = self._forward_eval(ipts)
        self._extract_and_update_metrics(result_set, ipts)
        # Return only what's needed for predictions/visualization (skip large intermediate tensors)
        keep_keys = {"refined_poses_xyz", "assignment_xyzs", "assignment_gt_xyzs", "root_xys"}
        retval = {k: v for k, v in result_set.items() if k in keep_keys}
        for k in ["gt_keypoints_xyzs", "img_paths", "cam_params_vec"]:
            retval[k] = ipts[k]
        retval["sequence"] = ipts["sequence"]
        return retval

    # ------------------------------------------------------------------
    # Input preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, batch_data: dict) -> dict:
        for k, v in batch_data.items():
            batch_data[k] = move_to_device(v, self.device)
        return batch_data

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _forward_train(self, batch: dict) -> dict:
        feat_output = self.backbone(batch)

        result_set = {
            "gt_keypoints_xyzs": batch["gt_keypoints_xyzs"],
            "gt_keypoints_xys": batch["gt_keypoints_xys"],
            "person_ids": batch["person_ids"],
            "affine_transforms": batch["affine_transforms"],
            "cam_params_vec": batch["cam_params_vec"],
        }

        has_assignment = hasattr(self, "root_regression_net")
        has_regressor = hasattr(self, "pose_regression_net")

        if not (has_assignment or has_regressor):
            result_set["heatmaps"] = feat_output["feature_maps"][0]
            return result_set

        result_set["heatmaps"] = feat_output["feature_maps"][0]   # supervised by Heatmap_Loss

        instance_dict = self.backbone.create_instances(batch, feat_output["heatmaps"], mode="train")  # type: ignore
        backbone_ret_val = {
            "feature_maps": feat_output["feature_maps"],
            "instance_dict": instance_dict,
        }

        # Assignment: train mode produces diffusion supervision tensors (x0_hat / u0_hat / ...)
        # but no decoded assignment XYZs (decoding is non-differentiable and lives in eval).
        # The regressor needs assignment XYZs to attend to, so for joint training we always
        # also compute the GT-derived pseudo reference points. This teacher-forces clean
        # assignments into the regressor while letting the assignment-net learn from its own
        # X_Loss / U_Loss — gradients don't flow between the two paths during training.
        assignment_ret: dict = {}
        if has_assignment and "assignment" not in self._freeze_list:
            assignment_ret = self.root_regression_net(
                feature_maps=backbone_ret_val["feature_maps"],
                instances_dict=backbone_ret_val["instance_dict"],
                cam_params_vec=batch["cam_params_vec"],
                projection_params={k: batch[k] for k in ("center", "scale", "rotation")},
                mode="train",
            )
        if has_regressor and "assignment_xyzs" not in assignment_ret:
            assignment_ret.update(
                self._get_pseudo_reference_points(backbone_ret_val, batch, mode="train")
            )

        result_set.update(assignment_ret)

        # Regression
        if has_regressor:
            result_set["regression_val"] = self.pose_regression_net(
                backbone_ret_val=backbone_ret_val,
                scale=batch["scale"],
                center=batch["center"],
                rotation=batch["rotation"],
                cam_params_vec=batch["cam_params_vec"],
                data=assignment_ret,
                mode="train",
            )

        return result_set

    @torch.no_grad()
    def _forward_eval(self, batch: dict) -> dict:
        result_set: dict[str, Any] = {}

        feat_output = self.backbone(batch)
        result_set["heatmaps"] = feat_output["feature_maps"][0]

        instance_dict = self.backbone.create_instances(batch, feat_output["heatmaps"], mode="test")  # type: ignore
        result_set["root_xys"] = instance_dict["image"][..., 2, :]

        backbone_ret_val = {
            "feature_maps": feat_output["feature_maps"],
            "instance_dict": instance_dict,
        }

        # Assignment
        if hasattr(self, "root_regression_net"):
            assignment_ret = self.root_regression_net(
                feature_maps=backbone_ret_val["feature_maps"],
                instances_dict=backbone_ret_val["instance_dict"],
                cam_params_vec=batch["cam_params_vec"],
                projection_params={k: batch[k] for k in ("center", "scale", "rotation")},
                mode="test",
            )
        elif hasattr(self, "pose_regression_net"):
            assignment_ret = self._get_pseudo_reference_points(backbone_ret_val, batch, mode="test")
        else:
            assignment_ret = {}

        result_set |= dict(assignment_ret)

        # Regression
        if hasattr(self, "pose_regression_net"):
            regression_ret = self.pose_regression_net(
                backbone_ret_val=backbone_ret_val,
                scale=batch["scale"],
                center=batch["center"],
                rotation=batch["rotation"],
                cam_params_vec=batch["cam_params_vec"],
                data=assignment_ret,
                mode="test",
            )
            result_set |= regression_ret

        # Alias prior_poses_xyz → refined_poses_xyz if no refinement
        if "refined_poses_xyz" not in result_set and "prior_poses_xyz" in result_set:
            result_set["refined_poses_xyz"] = result_set["prior_poses_xyz"]

        return result_set

    def _extract_and_update_metrics(self, result_set: dict, batch: dict) -> None:
        """Extract center-frame predictions from result_set and update metrics."""
        T = batch["cam_params_vec"].shape[2]
        t_mid = T // 2

        if "assignment_xyzs" in result_set:
            assignment_xyzs = result_set["assignment_xyzs"][:, 0, t_mid]
            assignment_xyz = assignment_xyzs[..., :3]
            assignment_s = assignment_xyzs[..., -1]
        else:
            assignment_xyzs = None
            assignment_xyz = None
            assignment_s = None

        assignment_gt_xyzs = result_set.get("assignment_gt_xyzs")
        if assignment_gt_xyzs is not None:
            assignment_gt_xyzs = assignment_gt_xyzs[:, 0, t_mid]

        pred_poses_xyz = None
        if "refined_poses_xyz" in result_set:
            pred_poses_xyz = result_set["refined_poses_xyz"][:, 0, t_mid, -1]

        self._update_metrics(
            batch=batch,
            heatmaps=result_set.get("heatmaps"),
            root_xys=result_set.get("root_xys"),
            pred_poses_xyz=pred_poses_xyz,
            assignment_xyz=assignment_xyz,
            assignment_xyzs=assignment_xyzs,
            assignment_s=assignment_s,
            assignment_gt_xyzs=assignment_gt_xyzs,
            global_ids=batch["global_ids"][:, 0, t_mid].long(),
            poses_xyzs_gt=batch["gt_keypoints_xyzs"][:, 0, t_mid],
        )

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _update_metrics(
        self,
        batch,
        heatmaps,
        root_xys,
        pred_poses_xyz,
        assignment_xyz,
        assignment_xyzs,
        assignment_s,
        assignment_gt_xyzs,
        global_ids,
        poses_xyzs_gt,
    ):
        for name, metric in self._metrics.named_children():
            match name:
                case "RegressionMetrics":
                    metric.update(
                        prediction=pred_poses_xyz,
                        confidence=assignment_s,
                        ground_truth=poses_xyzs_gt,
                        global_ids=global_ids,
                        gt_actor_indices=batch["person_ids"][:, 0, 0],
                    )
                case "PCPMetrics":
                    # PCP is the standard metric for Shelf / Campus.
                    # gt_actor_indices: (B, N_gt) actor index per GT pose;
                    # we use the first-view person_ids slice (constant across views
                    # by construction in the dataset loaders).
                    metric.update(
                        prediction=pred_poses_xyz,
                        ground_truth=poses_xyzs_gt,
                        gt_actor_indices=batch["person_ids"][:, 0, 0, 0],
                    )
                case "RegressionVisualizationMetrics":
                    metric.update(
                        regression_pred_xyz=pred_poses_xyz,
                        confidence=assignment_s,
                        regression_gt_xyzs=poses_xyzs_gt,
                    )
                case "HeatMapMetrics":
                    metric.update(
                        prediction=heatmaps,
                        gt_keypoints_xys=batch["gt_keypoints_xys"],
                        affine_transforms=batch["affine_transforms"],
                    )
                case "HeatMapRegressionMetrics":
                    metric.update(
                        prediction=root_xys,
                        ground_truth=batch["gt_keypoints_xys"][..., 2, :],
                    )
                case "RootMetrics":
                    metric.update(
                        prediction=assignment_xyz,
                        confidence=assignment_s,
                        global_ids=global_ids,
                        pseudo_ground_truth=assignment_gt_xyzs,
                        ground_truth=poses_xyzs_gt,
                    )

    def compute_metrics(self) -> dict:
        metrics_dict: dict[str, Any] = {}
        for name, metric in self._metrics.named_children():
            result = metric.compute()
            metrics_dict.update(
                {
                    f"{name}_{k.capitalize()}" if not k[0].isupper() else f"{name}_{k}": (
                        v.detach() if is_tensor(v) else v
                    )
                    for k, v in result.items()
                }
            )
        return metrics_dict

    def reset_metrics(self) -> None:
        for metric in self._metrics.children():
            metric.reset()

    # ------------------------------------------------------------------
    # Pseudo reference points (when no assignment net)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_pseudo_reference_points(self, backbone_ret_val, batch, mode: str):
        img_xy = backbone_ret_val["instance_dict"]["image"][..., 2, :]
        node_scores = backbone_ret_val["instance_dict"]["score"][..., 2, 0]
        person_ids = backbone_ret_val["instance_dict"]["person_ids"][..., 0]
        cam_params_vec = batch["cam_params_vec"]

        B, A, T, V, N_det = node_scores.shape
        pseudo_gt_person_ids = batch["person_ids"]
        N_slot = pseudo_gt_person_ids.shape[-1]

        det_pids_exp = person_ids.unsqueeze(-1)
        slot_pids_exp = pseudo_gt_person_ids.unsqueeze(4).long()
        match = (det_pids_exp == slot_pids_exp) & (det_pids_exp >= 0)
        has_match = match.any(dim=-1)
        slot_idx = match.float().argmax(dim=-1)

        img_xy_vessel = img_xy.new_zeros((B, A, T, V, N_slot, 3))
        gt_scores = node_scores.new_zeros((B, A, T, V, N_slot))

        b_idx, a_idx, t_idx, v_idx, n_det_idx = torch.where(has_match)
        n_slot_idx = slot_idx[b_idx, a_idx, t_idx, v_idx, n_det_idx]

        img_xy_vessel[b_idx, a_idx, t_idx, v_idx, n_slot_idx] = img_xy[b_idx, a_idx, t_idx, v_idx, n_det_idx]
        gt_scores[b_idx, a_idx, t_idx, v_idx, n_slot_idx] = node_scores[b_idx, a_idx, t_idx, v_idx, n_det_idx]

        img_xy_vessel = rearrange(img_xy_vessel, "b a t v n d -> v b a t n d")
        gt_scores = rearrange(gt_scores, "b a t v n -> v b a t n")
        cam_params_vec = repeat(cam_params_vec, "b a t v c -> v b a t n c", n=N_slot)

        assignment_xyz = triangulate_algebraic(img_xy_vessel[..., :2], cam_params_vec, gt_scores)

        assignment_valid = assignment_xyz.sum(-1, keepdim=True) != 0
        assignment_valid_f = assignment_valid.to(assignment_xyz.dtype)

        if mode == "train":
            valid_flat = assignment_valid.squeeze(-1)
            if valid_flat.any():
                noise = torch.randn_like(assignment_xyz[valid_flat]) * PSEUDO_REF_JITTER_MM
                assignment_xyz[valid_flat] = assignment_xyz[valid_flat] + noise

        assignment_xyzs = torch.cat([assignment_xyz, assignment_valid_f], dim=-1)

        invalid_mask = ~assignment_valid.squeeze(-1).unsqueeze(3).expand_as(batch["person_ids"])
        batch["person_ids"][invalid_mask] = -1

        return {"assignment_xyzs": assignment_xyzs}
