"""RootRegressionNet.

Composes the deformable-attention feature reader, the geometric edge-cue
module, the Sinkhorn solver, and the projected-flow / DDIM diffusion module.
All assignment-specific helpers live under ``src/models/assignment/``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from src.models.assignment.data import get_matching_data, match_gt_to_edges
from src.models.assignment.decode import (
    decode_and_triangulate,
    projected_mass_to_confidence,
    triangulate_gt,
)
from src.models.assignment.denoiser import GCNDenoiser
from src.models.assignment.edge_cues import GeometricHyperedgeCue
from src.models.assignment.graph import build_graph
from src.models.assignment.sampler import ProjectedConstrainedDDIM
from src.models.assignment.sinkhorn import SparseSinkhornSolver
from src.models.shared.proj_attention import ProjAttn
from src.utils.paramUtil import (
    CAMPUS_CAM_CONFIGURATIONS,
    MMOR_CAM_CONFIGURATIONS,
    PANOPTIC_CAM_CONFIGURATIONS,
    SHELF_CAM_CONFIGURATIONS,
)


_CAMERA_CONFIGS = {
    "CMU": PANOPTIC_CAM_CONFIGURATIONS,
    "Shelf": SHELF_CAM_CONFIGURATIONS,
    "Campus": CAMPUS_CAM_CONFIGURATIONS,
    "MMOR": MMOR_CAM_CONFIGURATIONS,
}


def _resolve_cameras(camera_setup: str) -> list[int]:
    for prefix, table in _CAMERA_CONFIGS.items():
        if camera_setup.startswith(prefix):
            return table[camera_setup]
    raise ValueError(f"Unknown camera setup: {camera_setup}")


class RootRegressionNet(nn.Module):
    """Projected-flow / DDIM cross-view assignment network.

    Outputs at inference (per ``forward(..., mode='test')``):
      - ``assignment_xyzs``: ``(B, A, T, N, 4)`` — decoded per-person ``(x, y, z, score)``.
      - ``assignment_gt_xyzs``: ``(B, A, T, N, 4)`` — GT triangulated for eval.
      - ``x0_hat``: ``(M,)`` — projected edge mass from the diffusion sampler.
    """

    # Fixed across all current tasks; kept as module-level constants
    _MATCHING_DROP_PROB = 0.2
    _E0_LOGIT_SCALE = 5.0

    def __init__(
        self,
        camera_setup: str,
        heatmap_size: tuple[int, int],
        alpha: float,
        px_threshold: int,
        num_timesteps: int,
        skh_iterations: int,
        sampling_timesteps: int,
        matcher_threshold: float,
        proj_attn_cfg: dict,
        denoiser_cfg: dict,
        use_heatmap_confidence: bool = True,
    ):
        super().__init__()

        cameras = _resolve_cameras(camera_setup)
        assert len(cameras) >= 2, f"Need at least 2 views, got {len(cameras)}"
        self.num_views = len(cameras)

        # Scalar settings used by forward + decode.
        self.heatmap_size = heatmap_size
        self.matcher_threshold = matcher_threshold
        self.use_heatmap_confidence = bool(use_heatmap_confidence)

        # Sinkhorn dustbin prior — frozen at the task-config value
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float64), requires_grad=False)

        # --- Denoiser network (edge state is a single-channel mass tensor) ---
        denoiser_cfg = dict(denoiser_cfg)
        denoiser_cfg["edge_in_dim"] = 1
        # ProjAttn outputs d_model features; keep the denoiser's node head aligned.
        denoiser_cfg.setdefault("node_in_dim", denoiser_cfg["d_model"])
        self.denoiser = GCNDenoiser(**denoiser_cfg)

        # --- Sinkhorn projection onto the polystochastic polytope ---
        self.sinkhorn_solver = SparseSinkhornSolver(num_views=self.num_views)

        # --- Deformable cross-view feature reader ---
        proj_attn_cfg = dict(proj_attn_cfg)
        proj_attn_cfg["d_model"] = denoiser_cfg["d_model"]
        proj_attn_cfg["heatmap_size"] = heatmap_size
        # Backbone deconv tail outputs 256-channel feature maps regardless of d_model.
        proj_attn_cfg.setdefault("feature_dim", 256)
        self.proj_attn = ProjAttn(**proj_attn_cfg)

        # --- Geometric edge cue ---
        self.z_cue = GeometricHyperedgeCue(num_views=self.num_views, lam=0.01, pixel_threshold=px_threshold**2)

        # --- Projected-flow / DDIM diffusion (wraps denoiser + sinkhorn + alpha) ---
        self.diffusion = ProjectedConstrainedDDIM(
            denoiser=self.denoiser,
            sinkhorn_solver=self.sinkhorn_solver,
            alpha_param=self.alpha,
            num_timesteps=num_timesteps,
            sampling_timesteps=sampling_timesteps,
            skh_iterations=skh_iterations,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        instances_dict: dict[str, torch.Tensor],
        cam_params_vec: torch.Tensor,
        projection_params: dict[str, torch.Tensor] | None = None,
        mode: str = "train",
    ) -> dict[str, torch.Tensor]:
        B, A, T, V, _ = cam_params_vec.shape
        Bp = B * A * T
        N = instances_dict["image"].shape[4]
        device = cam_params_vec.device

        feat_flat, inst_flat, cam_flat = self._flatten_batch_dims(
            feature_maps, instances_dict, cam_params_vec
        )
        proj_flat = (
            {k: rearrange(v, "b a t ... -> (b a t) ...") for k, v in projection_params.items()}
            if projection_params is not None
            else None
        )

        if mode == "train":
            train_data = get_matching_data(inst_flat, drop_prob=self._MATCHING_DROP_PROB)
            instances = train_data["instances"]
            node_mask: torch.Tensor = train_data["valid_mask"]  # type: ignore
            gt_correspondences: torch.Tensor = train_data["gt_correspondences"]  # type: ignore
        else:
            instances = {k: v for k, v in inst_flat.items() if k != "person_ids"}
            node_mask = (instances["score"] > 0).any(dim=(3, 4))
            gt_correspondences = None  # type: ignore

        graph = build_graph(
            feat_flat,
            instances,
            cam_flat,
            node_mask,
            proj_attn=self.proj_attn,
            z_cue=self.z_cue,
            heatmap_size=self.heatmap_size,
            d_model=self.denoiser.d_model,
        )

        # Empty-graph short-circuit.
        if graph["M"] == 0:
            if mode == "train":
                return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}
            return self._empty_result(B, A, T, N, device)

        # Training: GT-matched edge supervision.
        if mode == "train" and gt_correspondences is not None:
            E0 = match_gt_to_edges(gt_correspondences, graph["edge_tuples"])
            return self.diffusion.forward_train(
                graph=graph,
                node_mask=node_mask,
                E0_bin=E0,
                Bp=Bp,
                e0_logit_scale=self._E0_LOGIT_SCALE,
            )

        # Inference: sample → score → greedy decode → triangulate.
        x0_hat = self.diffusion.sample(graph=graph, node_mask=node_mask, Bp=Bp)
        E_scores = projected_mass_to_confidence(x0_hat, graph["z_cue"])

        assignment_xyz = decode_and_triangulate(
            E_scores=E_scores,
            edge_tuples=graph["edge_tuples"],
            inst_flat=inst_flat,
            cam_flat=cam_flat,
            heatmaps=feat_flat[0] if self.use_heatmap_confidence else None,
            projection_params=proj_flat,
            num_views=self.num_views,
            matcher_threshold=self.matcher_threshold,
            use_heatmap_confidence=self.use_heatmap_confidence,
        )

        person_ids = rearrange(instances_dict["person_ids"], "b a t ... -> (b a t) ...")
        assignment_gt_xyzs = triangulate_gt(
            person_ids,
            inst_flat["image"],
            inst_flat["score"],
            cam_flat,
            num_views=self.num_views,
        )

        return {
            "assignment_xyzs": rearrange(assignment_xyz, "(b a t) n d -> b a t n d", b=B, a=A, t=T),
            "assignment_gt_xyzs": rearrange(assignment_gt_xyzs, "(b a t) ... -> b a t ...", b=B, a=A, t=T)[..., 0, :],
            "x0_hat": x0_hat,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_batch_dims(
        self,
        feature_maps: list[torch.Tensor],
        instances_dict: dict[str, torch.Tensor],
        cam_params_vec: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Collapse ``(B, A, T, ...)`` → ``(B*A*T, ...)`` and select the root joint."""
        feat_flat = [rearrange(x, "b a t ... -> (b a t) ...") for x in feature_maps]

        inst_flat: dict[str, torch.Tensor] = {}
        for k, v in instances_dict.items():
            val = rearrange(v, "b a t ... -> (b a t) ...")
            if k != "person_ids" and val.shape[-2] > 1:
                val = val[..., 2:3, :]  # keep only root joint
            inst_flat[k] = val

        cam_flat = rearrange(cam_params_vec, "b a t ... -> (b a t) ...")
        return feat_flat, inst_flat, cam_flat

    def _empty_result(
        self, B: int, A: int, T: int, N: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Zero-shaped result when no candidate edges survive geometric filtering."""
        return {
            "assignment_xyzs": torch.zeros(B, A, T, N, 4, device=device),
            "assignment_gt_xyzs": torch.zeros(B, A, T, N, 4, device=device),
            "x0_hat": torch.zeros(0, device=device),
        }
