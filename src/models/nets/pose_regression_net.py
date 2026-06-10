"""
This class is responsible for regressing the 3D joint positions from the 3D centers
"""

import logging

import torch
import torch.nn as nn
from einops import rearrange, repeat

log = logging.getLogger(__name__)


class PoseRegressionNet(nn.Module):
    def __init__(
        self,
        space_configuration: dict,
        prior_module: nn.Module,
        refine_module: nn.Module,
        num_joints: int = 15,
        max_instances: int = 10,
        prior_noise_scale: float = 18.0,
        ckpt_path: str | None = None,
    ):
        super().__init__()
        self.J = num_joints
        self.N = max_instances
        self.prior_noise_scale = prior_noise_scale

        self.space_configuration = space_configuration

        self.prior_module = prior_module
        self.refine_module = refine_module

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", {}))
            filtered_state_dict = {
                k.replace("pose_regression_net.", ""): v
                for k, v in state_dict.items()
                if k.startswith("pose_regression_net.")
            }
            self.load_state_dict(filtered_state_dict, strict=True)
            log.info(f"Loaded GraphDecoder weights from {ckpt_path}")

    def forward(
        self,
        backbone_ret_val: dict,
        scale: torch.Tensor,
        center: torch.Tensor,
        rotation: torch.Tensor,
        cam_params_vec: torch.Tensor,
        data: dict,
        mode: str = "train",
    ):
        # list of shape (n_levels + joint_heatmap | (B, A, T, V, C, H, W))
        feature_maps = backbone_ret_val["feature_maps"]
        B, A, T, V, C = cam_params_vec.shape

        # (B, A, T, N, 4) - predicted root positions and score
        assignment_xyzs = data["assignment_xyzs"].detach()

        # Flatten over B, A, T
        flatten_fn = lambda x: rearrange(x, "b a t ... -> (b a t) ...")  # noqa: E731

        feature_maps_f = [flatten_fn(x) for x in feature_maps]
        scale_f = flatten_fn(scale)
        center_f = flatten_fn(center)
        rotation_f = flatten_fn(rotation)
        cam_params_vec_f = flatten_fn(cam_params_vec)
        assignment_xyzs_f = flatten_fn(assignment_xyzs)

        valid_mask = assignment_xyzs_f[..., -1] > 0  # (B * A * T, N) visibility mask
        valid_mask = repeat(valid_mask, "b n -> b n j", j=self.J)

        prior_ret_val = self.prior_module(
            assignment_xyzs_f,
            feature_maps_f[0],
            center_f,
            scale_f,
            rotation_f,
            cam_params_vec_f,
        )

        refine_ret_val = {}
        if self.refine_module is not None:
            prior_reference_poses_xyz = prior_ret_val["prior_poses_xyz"]

            # If the prior is learnable, jitter it during training only (keeps eval deterministic)
            if mode == "train" and any(p.requires_grad for p in self.prior_module.parameters()):
                noise = (2 * torch.rand_like(prior_ret_val["prior_poses_xyz"]) - 1) * self.prior_noise_scale
                prior_reference_poses_xyz = prior_reference_poses_xyz + noise

            refine_ret_val = self.refine_module(
                prior_reference_poses_xyz.detach(),
                feature_maps_f,
                scale_f,
                center_f,
                rotation_f,
                cam_params_vec_f,
                mask=valid_mask,
            )

        # Indices included for reshaping
        indices = dict(b=B, a=A, t=T, n=self.N, j=self.J)
        pattern = "(b a t) ... (n j) d -> b a t ... n j d"

        for key, value in refine_ret_val.items():
            refine_ret_val[key] = rearrange(value, pattern, **indices)

        pattern = "(b a t) ... n j d -> b a t ... n j d"
        for key, value in prior_ret_val.items():
            prior_ret_val[key] = rearrange(value, pattern, **indices)

        out = prior_ret_val | refine_ret_val

        return out
