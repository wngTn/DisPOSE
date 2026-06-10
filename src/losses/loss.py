from typing import Any

import torch
from einops import repeat
from torch import Tensor

from .base import BaseLosses
from .custom.pseudo_regression_loss import original_image_resolution, process_pseudo_2d


def reduce_person_ids_over_views(person_ids: torch.Tensor) -> torch.Tensor:
    """Reduce person_ids (B, A, T, V, N) over V dimension -> (B, A, T, N)."""
    valid_mask = person_ids != -1  # (B, A, T, V, N)

    big_val = (
        torch.finfo(person_ids.dtype).max
        if person_ids.is_floating_point()
        else torch.iinfo(person_ids.dtype).max
    )

    max_vals = person_ids.max(dim=-2).values  # (B, A, T, N)
    min_valid = person_ids.where(valid_mask, person_ids.new_tensor(big_val)).min(dim=-2).values

    has_valid = valid_mask.any(dim=-2)  # (B, A, T, N)
    if has_valid.any():
        if not (max_vals[has_valid] == min_valid[has_valid]).all():
            raise ValueError("Non -1 person IDs disagree across views!")

    return torch.where(has_valid, max_vals, person_ids.new_tensor(-1))


class Loss(BaseLosses):
    def __init__(
        self,
        cfg: Any,
        heatmap_supervise_unmatched: bool = True,
        image_size: tuple[int, int] | None = None,
        **kwargs,
    ):
        super().__init__(cfg, **kwargs)
        # Network input size (W, H); used to recover the dataset's original camera
        # resolution for the reprojection in-bounds check in process_pseudo_2d.
        self.image_size = image_size
        # If False, the Heatmap_Loss masks out 2D detections without a 3D
        # correspondence (person_id < 0). Useful for datasets where unmatched
        # detections are dominated by noise (reflections, equipment, etc.) —
        # e.g. MM-OR. Default True keeps all valid 2D points (see how invalid/unmatched
        # 2D coords are stored during label loading in panoptic_utils).
        self.heatmap_supervise_unmatched = bool(heatmap_supervise_unmatched)

    def update(
        self,
        ret_val: dict[str, Any],
    ) -> Tensor:
        """Update all relevant losses based on current stage and return total loss."""
        total = torch.tensor(0.0, device=self.total.device)  # type: ignore

        if "loss" in ret_val:
            total += ret_val["loss"]
            self.total.add_(ret_val["loss"].detach())
            self.count.add_(1)
            return total

        # --- Backbone heatmap supervision ---
        if "Heatmap_Loss" in self.losses:
            heatmap_pred = ret_val["heatmaps"]  # (*, C, H, W)
            affine_transforms = ret_val["affine_transforms"]  # (B, A, T, V, 2, 3)
            cam_params_vec = ret_val["cam_params_vec"]  # (B, A, T, V, 4)

            gt_poses_xyzs = ret_val["gt_keypoints_xyzs"]
            gt_poses_xys = ret_val["gt_keypoints_xys"]

            # By default we keep all valid 2D detections (HeatmapLoss internally
            # filters via `gt_score > 0`). If `heatmap_supervise_unmatched=False`
            # we additionally drop detections without a 3D correspondence
            # (person_id < 0) — useful when unmatched detections are mostly noise.
            if not self.heatmap_supervise_unmatched:
                person_ids = ret_val["person_ids"]
                gt_poses_xys = gt_poses_xys * (person_ids >= 0).unsqueeze(-1).unsqueeze(-1)

            total += self._update_loss(
                "Heatmap_Loss",
                heatmap_pred,
                gt_poses_xys,
                gt_poses_xyzs,
                cam_params_vec,
                affine_transforms,
            )

        # --- Assignment diffusion supervision ---
        if "x0_hat" in ret_val and "X_Loss" in self.losses:
            total += self._update_loss("X_Loss", ret_val["x0_hat"], ret_val["x0_gt"])
        if "u0_hat" in ret_val and "U_Loss" in self.losses:
            total += self._update_loss("U_Loss", ret_val["u0_hat"], ret_val["u0"])

        # --- Regressor refined-stage supervision ---
        if "regression_val" in ret_val:
            regression_ret_val = ret_val["regression_val"]

            person_ids = ret_val["person_ids"]
            gt_poses_xyzs = ret_val["gt_keypoints_xyzs"]
            gt_poses_xys = ret_val["gt_keypoints_xys"]

            prior_pred_2d_list = [
                regression_ret_val.get(k)
                for k in ["prior_reproj_poses_xy"]
                if k in regression_ret_val
            ]
            prior_pred_3d_list = [
                regression_ret_val.get(k) for k in ["prior_poses_xyz"] if k in regression_ret_val
            ]
            refined_pred_2d_list = [
                regression_ret_val.get(k)
                for k in ["refined_reproj_poses_xy"]
                if k in regression_ret_val
            ]
            refined_pred_3d_list = [
                regression_ret_val.get(k) for k in ["refined_poses_xyz"] if k in regression_ret_val
            ]

            gt_poses_xys = gt_poses_xys * (person_ids >= 0).unsqueeze(-1).unsqueeze(-1)
            img_width, img_height = (
                original_image_resolution(self.image_size) if self.image_size is not None else (1920, 1080)
            )
            gt_poses_xys = process_pseudo_2d(
                gt_poses_xyzs, gt_poses_xys, ret_val["cam_params_vec"], img_width=img_width, img_height=img_height
            )

            affine_transforms = ret_val["affine_transforms"]  # (B, A, T, V, 2, 3)

            try:
                L = regression_ret_val["refined_poses_xyz"].shape[3]
            except (KeyError, IndexError):
                L = 0
            triangulation_residuals = regression_ret_val.get("triangulation_residuals")
            triangulation_valid_mask = regression_ret_val.get("triangulation_valid_mask")

            prior_losses = {
                "PseudoRegressionCrossAffineL1Loss": [prior_pred_3d_list, person_ids],
                "Pseudo3DRegressionL1Loss": [prior_pred_3d_list, gt_poses_xyzs],
                "PseudoRegressionL1Loss": [prior_pred_2d_list, gt_poses_xys],
                "PseudoRegressionL2Loss": [prior_pred_2d_list, gt_poses_xys, affine_transforms],
            }
            refined_losses = {
                "TriangulationResidualLoss": [
                    [triangulation_residuals.squeeze(-1) if triangulation_residuals is not None else None],
                    triangulation_valid_mask.squeeze(-1) if triangulation_valid_mask is not None else None,
                ],
                "RefinedPseudoRegressionCrossAffineL1Loss": [
                    refined_pred_3d_list,
                    repeat(reduce_person_ids_over_views(person_ids), "b a t n -> b a t l n", l=L),
                ],
                # AnchorL1Loss: per-stage L1 between refined 3D prediction and GT 3D pose.
                # Tethers the refined output to GT geometry while reprojection losses shape it
                # against per-view 2D evidence.
                "AnchorL1Loss": [
                    refined_pred_3d_list,
                    repeat(gt_poses_xyzs, "b a t ... -> b a t l ...", l=L),
                ],
                "RefinedPseudoRegressionL1Loss": [
                    refined_pred_2d_list,
                    repeat(gt_poses_xys, "b a t ... -> b a t l ...", l=L),
                ],
                "RefinedPseudoRegressionL2Loss": [
                    refined_pred_2d_list,
                    repeat(gt_poses_xys, "b a t ... -> b a t l ...", l=L),
                    repeat(affine_transforms, "b a t ... -> b a t l ...", l=L),
                ],
            }

            for loss_key, args in prior_losses.items():
                if loss_key not in self.losses:
                    continue
                total += self._update_loss(loss_key, *args, dim_A=1)

            # Cross-stage cache for the L2 heatmap target (function of GT only — identical
            # for all deep-supervised stages). Computed lazily on the first registered
            # stage and reused for the rest.
            l2_target_cache = None
            l2_target_ready = False

            for l_idx in range(L):
                for loss_key, args in refined_losses.items():
                    full_key = f"{loss_key}_{l_idx}"
                    if full_key not in self.losses:
                        continue
                    args_0 = [x[:, :, :, l_idx] for x in args[0]]
                    args_1 = [x[:, :, :, l_idx] for x in args[1:]]
                    _args = [args_0, *args_1]

                    extra: dict = {"l_idx": l_idx, "dim_A": 1}
                    if loss_key == "RefinedPseudoRegressionL2Loss":
                        if not l2_target_ready:
                            l2_target_cache = self._losses_func[full_key].prepare_target(*args_1)
                            l2_target_ready = True
                        if l2_target_cache is None:
                            continue  # no valid joints — every stage would return 0
                        extra["target_cache"] = l2_target_cache

                    total += self._update_loss(full_key, *_args, **extra)

        self.total.add_(total.detach())  # type: ignore
        self.count.add_(1)  # type: ignore

        return total
