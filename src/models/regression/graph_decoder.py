import math
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from torch_geometric.nn.conv import HypergraphConv

from src.models.shared.proj_attention import ProjAttn
from src.models.shared.triangulation import AlgebraicRayTriangulator
from src.utils.camera import get_camera_params, world_3d_to_img_2d
from src.utils.linear_algebra import (
    affine_transform,
    get_affine_transform,
    transform_to_original_img_space,
)
from src.utils.paramUtil import (
    CAMPUS_CAM_CONFIGURATIONS,
    JOINT_PART_IDS,
    KEYPOINT_INFO,
    PANOPTIC_CAM_CONFIGURATIONS,
    SHELF_CAM_CONFIGURATIONS,
    MMOR_CAM_CONFIGURATIONS
)

NON_LINEAR_DICT: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "mish": nn.Mish,
    "elu": nn.ELU,
}


def init_linear_small(m: nn.Linear, std: float = 1e-3) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Rotary/Sinusoidal embedding that handles any integer value
    without a pre-defined maximum vocabulary size.
    """

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, ...] of integers or floats
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        x_scaled = x.unsqueeze(-1) * self.scale
        emb = x_scaled * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class OffsetNet2D(nn.Module):
    """
    Predicts per-view 2D offsets, triangulation weights, and visibility gates.
    """

    def __init__(
        self,
        d_model: int,
        activation: str,
        drop: float,
        max_offset_norm: float,
        img_size: Tuple[int, int],
    ):
        super().__init__()
        self.max_offset_norm = max_offset_norm
        self.register_buffer("img_size", torch.tensor(img_size, dtype=torch.float32), persistent=False)

        act_layer = NON_LINEAR_DICT.get(activation, nn.GELU)
        self.norm = nn.LayerNorm(d_model)
        self.trunk = nn.Sequential(
            Mlp(d_model, d_model, act_layer=act_layer, drop=drop),
        )
        self.offset_head = nn.Linear(d_model, 2)
        self.weight_head = nn.Linear(d_model, 1)

        # Conservative init: start with ~0 offset, ~uniform weights, ~mostly-open gates
        init_linear_small(self.offset_head, std=1e-3)
        init_linear_small(self.weight_head, std=1e-3)

        # Bias the visibility gates open at init (sigmoid(2) ~= 0.88) so valid views are not suppressed early in training.
        nn.init.constant_(self.weight_head.bias, 2.0)

    def forward(
        self,
        features: torch.Tensor,
        reference_xy_norm: torch.Tensor,
        valid_views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Residual connection
        h = self.norm(features)
        h = features + self.trunk(h)

        offset_raw = self.offset_head(h)  # [B, V, NJ, 2]
        weight_logits = self.weight_head(h).squeeze(-1)  # [B, V, NJ]

        # 1. Coordinate Refinement
        img_size_ex = self.img_size.view(1, 1, 1, 2)
        xy_pixels = reference_xy_norm * img_size_ex
        max_dim = self.img_size.max()

        offset_pixels = torch.tanh(offset_raw) * (max_dim * self.max_offset_norm)
        refined_xy = xy_pixels + offset_pixels

        # 2. Triangulation Weights (Softmax - Relative Confidence)
        masked_logits_softmax = weight_logits.masked_fill(~valid_views, -1e4)
        weights = F.softmax(masked_logits_softmax, dim=1)
        weights = weights * valid_views.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # 3. Visibility Gates (Sigmoid - Absolute Confidence)
        # Used to gate features entering the graph.
        # High logit -> 1.0 (Valid), Low logit -> 0.0 (Occluded/Noise)
        masked_logits_sigmoid = weight_logits.masked_fill(~valid_views, -1e4)
        gates = torch.sigmoid(masked_logits_sigmoid)

        return refined_xy, weights, gates


class ViewAggregator(nn.Module):
    def __init__(self, d_model: int, geo_gamma: float = 0.5, eps: float = 1e-8, detach_geo: bool = True):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, 1)
        self.geo_gamma = geo_gamma
        self.eps = eps
        self.detach_geo = detach_geo

    def forward(
        self, features: torch.Tensor, valid_views: torch.Tensor, geometry_weights: torch.Tensor
    ) -> torch.Tensor:
        # features: [B, V, NJ, C]
        # valid_views: [B, V, NJ]
        # geometry_weights: [B, V, NJ] (already normalized across V where valid)

        attn_logits = self.attn_proj(features).squeeze(-1)  # [B, V, NJ]
        attn_logits = attn_logits.masked_fill(~valid_views, -1e9)

        if geometry_weights is not None:
            w = geometry_weights
            if self.detach_geo:
                w = w.detach()
            w = w.clamp_min(self.eps)
            attn_logits = attn_logits + self.geo_gamma * torch.log(w)

        attn = F.softmax(attn_logits, dim=1)
        attn = attn * valid_views.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(self.eps)

        aggregated = (features * attn.unsqueeze(-1)).sum(dim=1)  # [B, NJ, C]
        has_any_view = valid_views.any(dim=1).unsqueeze(-1).float()
        return aggregated * has_any_view


class GraphDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        img_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        space_configuration: dict,
        attention_mode_stage1: str,
        attention_mode_stage2: str,
        n_points_proj: int,
        drop: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "gelu",
        w_forward_ffn: bool = True,
        V: int = 5,
        N: int = 10,
        max_offset_norm: float = 0.15,
        skeleton_type: str = "panoptic",
    ):
        super().__init__()
        self.V = V
        self.N = N
        self.w_forward_ffn = w_forward_ffn
        self.skeleton_type = skeleton_type

        # Constants setup
        self.J = len(KEYPOINT_INFO[skeleton_type])
        part_ids = torch.tensor(JOINT_PART_IDS[skeleton_type], dtype=torch.long)
        self.num_parts = int(part_ids.max().item()) + 1

        self._register_constants(space_configuration, img_size, part_ids)
        self._init_stage1_indices_template()
        self._init_stage2_indices_template()

        # Components
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = ProjAttn(d_model, n_heads, n_points_proj, heatmap_size)

        self.coordinates_linear = nn.Linear(3, d_model)
        self.joint_type_linear = nn.Linear(self.J, d_model)

        # Ray Encoder: Embeds 3D Ray Direction (World Space)
        self.ray_encoder = nn.Sequential(
            nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        self.stage1_res_scale = nn.Parameter(torch.tensor(0.0))
        self.stage2_res_scale = nn.Parameter(torch.tensor(0.0))

        # --- Stage 1: View-Joint Hypergraph ---
        self.stage1_hg = HypergraphConv(
            in_channels=d_model,
            out_channels=d_model,
            use_attention=True,
            attention_mode=attention_mode_stage1,
            heads=4,
            concat=False,
            dropout=drop,
        )
        self.stage1_norm = nn.LayerNorm(d_model)
        self.stage1_dropout = nn.Dropout(drop)

        # --- Offset Refinement ---
        self.offset_net = OffsetNet2D(
            d_model=d_model,
            activation=activation,
            drop=drop,
            max_offset_norm=max_offset_norm,
            img_size=img_size,
        )
        self.triangulator = AlgebraicRayTriangulator()

        # --- Stage 2: Person-Part Hypergraph ---
        self.view_agg = ViewAggregator(d_model)
        self.stage2_hg = HypergraphConv(
            in_channels=d_model,
            out_channels=d_model,
            use_attention=True,
            attention_mode=attention_mode_stage2,
            heads=4,
            concat=False,
            dropout=drop,
        )
        self.stage2_norm = nn.LayerNorm(d_model)
        self.stage2_dropout = nn.Dropout(drop)

        # --- Update Block ---
        self.feat_upd_norm = nn.LayerNorm(d_model)

        # Confidence Embedding
        self.view_count_embed = SinusoidalPositionalEmbedding(d_model // 4)

        # Geometric Feedback Encoder
        self.geo_feedback_dim = d_model // 2
        self.geo_feedback_mlp = nn.Sequential(
            nn.Linear(3 + 1 + (d_model // 4), self.geo_feedback_dim),
            nn.LayerNorm(self.geo_feedback_dim),
            nn.GELU(),
            nn.Linear(self.geo_feedback_dim, self.geo_feedback_dim),
        )

        # Zero-init
        nn.init.zeros_(self.geo_feedback_mlp[-1].weight)
        nn.init.zeros_(self.geo_feedback_mlp[-1].bias)

        self.feat_upd_linear = nn.Linear(d_model + self.geo_feedback_dim, d_model)
        self.feat_upd_dropout = nn.Dropout(drop)
        self.feat_upd_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.w_forward_ffn:
            act_layer = NON_LINEAR_DICT.get(activation, nn.GELU)
            self.ffn_norm = nn.LayerNorm(d_model)
            self.ffn_mlp = Mlp(d_model, d_ffn, d_model, act_layer=act_layer, drop=drop)
            self.ffn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # --- Helpers ---
    def _register_constants(self, space_cfg, img_size, part_ids):
        space_center = torch.tensor(space_cfg["space_center"], dtype=torch.float32)
        space_size = torch.tensor(space_cfg["space_size"], dtype=torch.float32)
        self.register_buffer("space_size", space_size, persistent=False)
        self.register_buffer("space_corner", space_center - 0.5 * space_size, persistent=False)
        self.register_buffer("img_size", torch.tensor(img_size, dtype=torch.float32), persistent=False)
        self.register_buffer("part_ids", part_ids, persistent=False)
        joint_types = F.one_hot(torch.arange(self.J), num_classes=self.J).float()
        self.register_buffer("joint_types_onehot", joint_types, persistent=False)

    def _normalize_coordinate(self, xyz: torch.Tensor) -> torch.Tensor:
        shape_diff = xyz.ndim - 1
        corner = self.space_corner.view(*([1] * shape_diff), 3)
        size = self.space_size.view(*([1] * shape_diff), 3)
        return ((xyz - corner) / size).clamp_(0.0, 1.0)

    # --- Graph Indices ---
    def _init_stage1_indices_template(self):
        num_nodes = self.V * self.N * self.J
        node_idx = torch.arange(num_nodes)
        j_idx = node_idx % self.J
        n_idx = (node_idx // self.J) % self.N
        v_idx = (node_idx // (self.N * self.J)) % self.V

        edge_intra = v_idx * self.N + n_idx
        num_intra = self.V * self.N
        edge_inter = num_intra + (n_idx * self.J + j_idx)

        nodes_all = repeat(node_idx, "m -> (2 m)")
        edges_all = torch.cat([edge_intra, edge_inter])

        self.register_buffer("s1_template_nodes", nodes_all, persistent=False)
        self.register_buffer("s1_template_edges", edges_all, persistent=False)
        self.s1_num_edges_per_batch = num_intra + (self.N * self.J)
        self.s1_num_nodes_per_batch = num_nodes

    def _init_stage2_indices_template(self):
        num_nodes = self.N * self.J
        node_idx = torch.arange(num_nodes)
        j_idx = node_idx % self.J
        n_idx = (node_idx // self.J) % self.N

        edge_person = n_idx
        part_id = self.part_ids[j_idx]
        num_person_edges = self.N
        edge_part = num_person_edges + (n_idx * self.num_parts + part_id)

        nodes_all = repeat(node_idx, "m -> (2 m)")
        edges_all = torch.cat([edge_person, edge_part])

        self.register_buffer("s2_template_nodes", nodes_all, persistent=False)
        self.register_buffer("s2_template_edges", edges_all, persistent=False)
        self.s2_num_edges_per_batch = num_person_edges + (self.N * self.num_parts)
        self.s2_num_nodes_per_batch = num_nodes

    def _get_stage1_indices(self, valid_views: torch.Tensor):
        B = valid_views.shape[0]
        device = valid_views.device
        batch_offsets_node = torch.arange(B, device=device).view(-1, 1) * self.s1_num_nodes_per_batch
        nodes = self.s1_template_nodes.unsqueeze(0).repeat(B, 1) + batch_offsets_node
        nodes = nodes.flatten()
        batch_offsets_edge = torch.arange(B, device=device).view(-1, 1) * self.s1_num_edges_per_batch
        edges = self.s1_template_edges.unsqueeze(0).repeat(B, 1) + batch_offsets_edge
        edges = edges.flatten()
        mask_nodes = valid_views.reshape(-1)
        mask_edges = repeat(mask_nodes, "m -> (2 m)")
        return torch.stack([nodes[mask_edges], edges[mask_edges]])

    def _get_stage2_indices(self, valid_joints: torch.Tensor):
        B = valid_joints.shape[0]
        device = valid_joints.device
        batch_offsets_node = torch.arange(B, device=device).view(-1, 1) * self.s2_num_nodes_per_batch
        nodes = self.s2_template_nodes.unsqueeze(0).repeat(B, 1) + batch_offsets_node
        nodes = nodes.flatten()
        batch_offsets_edge = torch.arange(B, device=device).view(-1, 1) * self.s2_num_edges_per_batch
        edges = self.s2_template_edges.unsqueeze(0).repeat(B, 1) + batch_offsets_edge
        edges = edges.flatten()
        mask_nodes = valid_joints.reshape(-1)
        mask_edges = repeat(mask_nodes, "m -> (2 m)")
        return torch.stack([nodes[mask_edges], edges[mask_edges]])

    def _project_and_mask(self, poses_xyz, cam_params, center, scale, rotation, mask):
        poses_xyz = rearrange(poses_xyz, "b (n j) d -> b n j d", n=self.N, j=self.J)
        xy_img, xy_valid = world_3d_to_img_2d(poses_xyz, cam_params)

        img_wh = center * 2
        inside_w = (xy_img[..., 0] >= 0) & (xy_img[..., 0] < img_wh[:, :, 0, None, None])
        inside_h = (xy_img[..., 1] >= 0) & (xy_img[..., 1] < img_wh[:, :, 1, None, None])
        inside = inside_w & inside_h & xy_valid.squeeze(-1)

        xy_img_clamped = torch.min(xy_img.clamp(min=-1.0), img_wh[..., None, None, :] - 1.0)
        affine = get_affine_transform(center, scale, rotation, self.img_size)

        xy_flat = xy_img_clamped.flatten(0, 1)
        aff_flat = affine.flatten(0, 1)
        xy_inp = affine_transform(xy_flat, aff_flat).view_as(xy_img)
        xy_norm_inp = xy_inp / self.img_size

        mask_nj = rearrange(mask, "b n j -> b 1 n j")
        valid_views = inside & mask_nj
        return xy_img, xy_norm_inp, valid_views

    def _get_stage1_attributes(self, B, joint_embed, instance_embed):
        num_intra = B * self.V * self.N
        n_ids = torch.arange(num_intra, device=instance_embed.device) % self.N
        attr_intra = instance_embed[n_ids]
        num_inter = B * self.N * self.J
        j_ids = torch.arange(num_inter, device=joint_embed.device) % self.J
        attr_inter = joint_embed[j_ids]
        return torch.cat([attr_intra, attr_inter])

    def _get_stage2_attributes(self, B, part_embed, instance_embed):
        num_person = B * self.N
        n_ids = torch.arange(num_person, device=instance_embed.device) % self.N
        attr_person = instance_embed[n_ids]
        num_part = B * self.N * self.num_parts
        p_ids = torch.arange(num_part, device=part_embed.device) % self.num_parts
        attr_part = part_embed[p_ids]
        return torch.cat([attr_person, attr_part])

    # --------------------------------------------------------------------------
    # Main Forward
    # --------------------------------------------------------------------------
    def forward(
        self,
        target: torch.Tensor,
        query: torch.Tensor,
        poses_xyz: torch.Tensor,
        feature_maps: List[torch.Tensor],
        scale: torch.Tensor,
        center: torch.Tensor,
        rotation: torch.Tensor,
        cam_params_vec: torch.Tensor,
        mask: torch.Tensor,
        joint_embed: torch.Tensor,
        instance_embed: torch.Tensor,
        part_embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, NJ, _ = poses_xyz.shape
        N = self.N
        J = self.J
        d_half = joint_embed.shape[-1] // 2

        # ----------------------------------------------------------------------
        # 1) Geometry projection (sampling)
        # ----------------------------------------------------------------------
        _, poses_xy_norm, valid_views_bool = self._project_and_mask(
            poses_xyz, cam_params_vec, center, scale, rotation, mask
        )

        q = self.cross_attn_norm(target + query if query is not None else target)
        mv_features = self.cross_attn(q, poses_xy_norm, feature_maps)

        # Zero out invalid features early
        mv_features = mv_features * valid_views_bool.unsqueeze(-1)

        # Flatten N,J into NJ for stage-1 convenience
        mv_features = mv_features.flatten(2, 3)  # [B, V, NJ, C]
        poses_xy_norm = poses_xy_norm.flatten(2, 3)  # [B, V, NJ, 2]
        valid_views_bool = valid_views_bool.flatten(2, 3)  # [B, V, NJ]

        # ----------------------------------------------------------------------
        # 2) Embeddings (position, joint-type, ray)
        # ----------------------------------------------------------------------
        # A) 3D position embedding (normalized to workspace)
        xyz_norm = self._normalize_coordinate(poses_xyz)  # [B, NJ, 3]
        pos_emb = self.coordinates_linear(xyz_norm)  # [B, NJ, C]
        pos_emb = rearrange(pos_emb, "b nj c -> b 1 nj c")

        # B) Joint-type embedding
        j_type_emb = self.joint_type_linear(self.joint_types_onehot)  # [J, C]
        j_type_emb = j_type_emb.view(1, 1, 1, J, -1).repeat(1, 1, N, 1, 1).view(1, 1, N * J, -1)

        # C) Ray embedding (camera geometry)
        (T_c2w, R_w2c) = get_camera_params(cam_params_vec, ["T_c2w", "R_w2c"])

        ray_diff_world = poses_xyz.unsqueeze(1) - T_c2w.unsqueeze(2)  # [B, V, NJ, 3]
        rays_cam_unnorm = torch.matmul(R_w2c.unsqueeze(2), ray_diff_world.unsqueeze(-1)).squeeze(-1)
        rays_cam = F.normalize(rays_cam_unnorm, dim=-1)
        ray_emb = self.ray_encoder(rays_cam)  # [B, V, NJ, C]

        # Add all embeddings and keep masked
        mv_features = mv_features + pos_emb + j_type_emb + ray_emb
        mv_features = mv_features * valid_views_bool.unsqueeze(-1)

        # ----------------------------------------------------------------------
        # 3) Stage 1: View-Joint hypergraph
        #    Refine per-(view, joint) features before predicting offsets/weights.
        # ----------------------------------------------------------------------
        x_in_flat = mv_features.view(-1, mv_features.shape[-1])  # [B*V*NJ, C]
        edge_idx_s1 = self._get_stage1_indices(valid_views_bool)

        h_attr_s1 = self._get_stage1_attributes(B, joint_embed[:, d_half:], instance_embed[:, d_half:])

        if edge_idx_s1.shape[1] > 0:
            x_out = self.stage1_hg(x_in_flat, edge_idx_s1, hyperedge_attr=h_attr_s1)
            x_out = self.stage1_norm(x_out)

            mask_flat = valid_views_bool.reshape(-1, 1).float()
            x_in_flat = x_in_flat + self.stage1_res_scale * self.stage1_dropout(x_out) * mask_flat

        mv_features = x_in_flat.view(B, self.V, N * J, -1)  # [B, V, NJ, C]

        # ----------------------------------------------------------------------
        # 4) Offset refinement & per-view weights/gates
        # ----------------------------------------------------------------------
        refined_xy_pixels, weights, gates = self.offset_net(mv_features, poses_xy_norm, valid_views_bool)

        # Use gates to suppress noisy views BEFORE view aggregation / stage 2
        mv_features = mv_features * gates.unsqueeze(-1)

        # ----------------------------------------------------------------------
        # 5) Triangulation
        # ----------------------------------------------------------------------
        refined_xy_orig = transform_to_original_img_space(
            refined_xy_pixels, center, scale, rotation[..., 0], self.img_size
        )

        refined_xy_flat = rearrange(refined_xy_orig, "b v nj c -> v (b nj) c")
        weights_flat = rearrange(weights, "b v nj -> v (b nj)")
        cams_flat = repeat(cam_params_vec, "b v c -> v (b nj) c", nj=N * J)

        valid_tri_mask = (valid_views_bool.sum(dim=1) >= 2) & mask.reshape(B, N * J)
        valid_tri_flat = valid_tri_mask.view(-1)

        refined_xyz_flat = poses_xyz.clone().view(-1, 3)
        residuals_flat = torch.zeros(B * N * J, device=poses_xyz.device)

        idx = valid_tri_flat.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            xyz_sel, res_sel = self.triangulator(refined_xy_flat[:, idx], cams_flat[:, idx], weights_flat[:, idx])
            refined_xyz_flat[idx] = xyz_sel[:, :3]
            residuals_flat[idx] = res_sel

        refined_xyz = refined_xyz_flat.view(B, N, J, 3)
        residuals = residuals_flat.view(B, N, J, 1)

        # ----------------------------------------------------------------------
        # 6) Stage 2: Person-Part hypergraph
        # ----------------------------------------------------------------------
        pose_feats = self.view_agg(mv_features, valid_views_bool, geometry_weights=weights)  # [B, NJ, C]
        pose_feats = pose_feats.view(B, N, J, -1)

        # Validity is defined purely from the visibility mask; the soft gates do not affect which joints are considered valid.
        valid_joints = valid_views_bool.any(dim=1).view(B, N, J)

        x_in2_flat = pose_feats.view(-1, pose_feats.shape[-1])
        edge_idx_s2 = self._get_stage2_indices(valid_joints)
        h_attr_s2 = self._get_stage2_attributes(B, part_embed[:, d_half:], instance_embed[:, d_half:])

        if edge_idx_s2.shape[1] > 0:
            x_out2 = self.stage2_hg(x_in2_flat, edge_idx_s2, hyperedge_attr=h_attr_s2)
            x_out2 = self.stage2_norm(x_out2)

            mask_flat_s2 = valid_joints.reshape(-1, 1).float()
            x_in2_flat = x_in2_flat + self.stage2_res_scale * self.stage2_dropout(x_out2) * mask_flat_s2

        pose_feats = x_in2_flat.view(B, N, J, -1)

        # ----------------------------------------------------------------------
        # 7) Feature update block
        # ----------------------------------------------------------------------
        xyz_fb = rearrange(refined_xyz, "b n j d -> b (n j) d").detach()
        res_fb = rearrange(residuals, "b n j d -> b (n j) d").detach()

        xyz_norm_feedback = self._normalize_coordinate(xyz_fb)

        is_triangulated = valid_tri_mask.view(B, N * J, 1)
        safe_residuals = torch.where(
            is_triangulated, res_fb.clamp(min=1e-6, max=1.0), torch.tensor(1.0, device=res_fb.device)
        )
        res_log = torch.log(safe_residuals)

        view_counts = valid_views_bool.sum(dim=1).long().clamp(max=self.V)
        count_emb = self.view_count_embed(view_counts)

        geo_in = torch.cat([xyz_norm_feedback, res_log, count_emb], dim=-1)
        geo_emb = self.geo_feedback_mlp(geo_in)

        pose_feats_flat = rearrange(pose_feats, "b n j c -> b (n j) c")
        norm_feats = self.feat_upd_norm(pose_feats_flat)
        feats_combined = torch.cat([norm_feats, geo_emb], dim=-1)

        t_out = self.feat_upd_dropout(self.feat_upd_linear(feats_combined))
        target = target + self.feat_upd_drop_path(t_out)

        if self.w_forward_ffn:
            target = target + self.ffn_drop_path(self.ffn_mlp(self.ffn_norm(target)))

        # ----------------------------------------------------------------------
        # 8) Output
        # ----------------------------------------------------------------------
        refined_reproj, refined_reproj_valid = world_3d_to_img_2d(refined_xyz, cam_params_vec)

        refined_xyz = rearrange(refined_xyz, "b n j d -> b (n j) d")
        refined_reproj = rearrange(refined_reproj, "b v n j d -> b v (n j) d")
        residuals = rearrange(residuals, "b n j d -> b (n j) d")
        valid_tri_mask_out = rearrange(valid_tri_mask, "b nj -> b nj 1")

        return {
            "target": target,
            "refined_poses_xyz": refined_xyz,
            "refined_reproj_poses_xy": refined_reproj,
            "refined_poses_xy": refined_xy_orig,
            "triangulation_residuals": residuals,
            "triangulation_valid_mask": valid_tri_mask_out,
        }


class GraphDecoder(nn.Module):
    """
    Multi-layer graph decoder.
    Refines 3D poses iteratively using Multi-view and Person-Part relationships.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        img_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        space_configuration: dict,
        layers_attn_modes: list[tuple[str, str]],
        camera_setup: str,
        max_instances: int = 10,
        return_intermediate: bool = True,
        skeleton_type: str = "panoptic",
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        if camera_setup.startswith("CMU"):
            cameras = PANOPTIC_CAM_CONFIGURATIONS[camera_setup]
        elif camera_setup.startswith("Shelf"):
            cameras = SHELF_CAM_CONFIGURATIONS[camera_setup]
        elif camera_setup.startswith("Campus"):
            cameras = CAMPUS_CAM_CONFIGURATIONS[camera_setup]
        elif camera_setup.startswith("MMOR"):
            cameras = MMOR_CAM_CONFIGURATIONS[camera_setup]
        else:
            raise ValueError(f"Unknown camera setup: {camera_setup}")
        assert len(cameras) >= 2, f"Need at least 2 views, got {len(cameras)}"

        self.d_model = d_model
        self.max_instances = max_instances
        self.return_intermediate = return_intermediate
        self.skeleton_type = skeleton_type
        self.J = len(KEYPOINT_INFO[skeleton_type])
        self.cameras = cameras

        self.joint_embedding = nn.Embedding(self.J, d_model * 2)
        self.instance_embedding = nn.Embedding(max_instances, d_model * 2)
        num_parts = max(JOINT_PART_IDS[skeleton_type]) + 1
        self.part_embedding = nn.Embedding(num_parts, d_model * 2)

        assert len(layers_attn_modes) > 0, "layers_attn_modes must be non-empty"
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(layers_attn_modes))]
        self.layers = nn.ModuleList()
        for i, (attn_mode1, attn_mode2) in enumerate(layers_attn_modes):
            self.layers.append(
                GraphDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ffn=d_ffn,
                    img_size=img_size,
                    heatmap_size=heatmap_size,
                    space_configuration=space_configuration,
                    attention_mode_stage1=attn_mode1,
                    attention_mode_stage2=attn_mode2,
                    n_points_proj=8,
                    drop=dropout,
                    drop_path=dpr[i],
                    activation=activation,
                    V=len(cameras),
                    N=max_instances,
                    skeleton_type=skeleton_type,
                )
            )

    def _get_query_target(self, B: int, N: int):
        joint_emb = self.joint_embedding.weight
        inst_emb = self.instance_embedding.weight[:N]

        combined = joint_emb.unsqueeze(0) + inst_emb.unsqueeze(1)
        combined = combined.reshape(N * self.J, -1)

        query, target = combined.chunk(2, dim=-1)

        query = repeat(query, "nj c -> b nj c", b=B)
        target = repeat(target, "nj c -> b nj c", b=B)
        return query, target

    def forward(
        self,
        poses_xyz: torch.Tensor,
        feature_maps: List[torch.Tensor],
        scale: torch.Tensor,
        center: torch.Tensor,
        rotation: torch.Tensor,
        cam_params_vec: torch.Tensor,
        mask: torch.Tensor,
    ) -> Union[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
        B, N, J = poses_xyz.shape[:3]
        mask = mask.bool()

        query, target = self._get_query_target(B, N)
        intermediates = defaultdict(list)
        current_poses = poses_xyz.flatten(1, 2)

        for layer in self.layers:
            out = layer(
                target=target,
                query=query,
                poses_xyz=current_poses,
                feature_maps=feature_maps,
                scale=scale,
                center=center,
                rotation=rotation,
                cam_params_vec=cam_params_vec,
                mask=mask,
                joint_embed=self.joint_embedding.weight,
                instance_embed=self.instance_embedding.weight,
                part_embed=self.part_embedding.weight,
            )
            target = out["target"]
            current_poses = out["refined_poses_xyz"].detach()

            if self.return_intermediate:
                for k, v in out.items():
                    intermediates[k].append(v)
            else:
                final_out = out

        if self.return_intermediate:
            return {k: torch.stack(v, dim=1) for k, v in intermediates.items()}

        return final_out
