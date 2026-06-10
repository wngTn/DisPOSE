import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import constant_, xavier_uniform_

from src.models.shared.deform_func import DeformFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class ProjAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_points: int,
        heatmap_size: tuple[int, int],
        sampling_mode: str = "deformable",
        feature_dim: int | None = None,
    ):
        """
        Projective Attention Module
        :param d_model      hidden dimension
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        :param heatmap_size (W, H) of the base-level feature map
        :param feature_dim  channel dim of the input feature maps; defaults to ``d_model``.
                            Set to the backbone channel count when it differs (e.g. d_model=128
                            with a 256-channel ResNet backbone).
        """
        super().__init__()
        if feature_dim is None:
            feature_dim = d_model
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in Deform to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation."
            )

        # Three-level feature pyramid derived from the base heatmap size (H, H/2, H/4).
        # heatmap_size: (W, H)
        _shapes = torch.tensor(
            [
                [heatmap_size[1], heatmap_size[0]],
                [heatmap_size[1] // 2, heatmap_size[0] // 2],
                [heatmap_size[1] // 4, heatmap_size[0] // 4],
            ],
            dtype=torch.long,
        )
        _sizes = _shapes[:, 0] * _shapes[:, 1]
        _cumsum = _sizes.cumsum(0)[:-1]
        _start_index = torch.cat((torch.zeros(1, dtype=torch.long), _cumsum))
        _offset_normalizer = _shapes.flip(-1).float()

        # Register them so they automatically move to GPU with the model
        self.register_buffer("spatial_shapes", _shapes, persistent=False)  # (3, 2) [H, W]
        self.register_buffer("feature_map_start_index", _start_index, persistent=False)  # (3,)
        self.register_buffer("offset_normalizer", _offset_normalizer, persistent=False)  # (3, 2) [W, H]

        self.n_levels = _shapes.shape[0]  # 3

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        if sampling_mode not in {"deformable", "reference", "mid_level"}:
            raise ValueError(f"Unknown ProjAttn sampling_mode: {sampling_mode}")
        self.sampling_mode = sampling_mode

        # --- Linear Layers ----
        # both weights and offsets are calculated from d_model, with a linear matrix (with learnt weights)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)

        # When the backbone feature width differs from d_model, pre-project each
        # feature level to d_model with a 1x1 conv before sampling. After this point
        # `reference_point_features` and `feature_maps_f` are both d_model-dim, so the
        # downstream sampling_offsets / attention_weights / value_proj / output_proj all
        # stay at d_model. With feature_dim == d_model this is a no-op (Identity).
        self.feature_dim = feature_dim
        self.feature_proj = (
            nn.Identity() if feature_dim == d_model else nn.Conv2d(feature_dim, d_model, kernel_size=1)
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # (H, 2)

        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]  # (H, 2)
        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)  # (H, P, 2)

        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor | None,  # (B, NJ, D)
        poses_xy_norm: torch.Tensor,  # (B, V, N, J, D) in [0, 1]
        feature_maps: list[torch.Tensor],  # list of (B, V, C, H_l, W_l)
    ):
        # Drop level 0
        feature_maps = feature_maps[1:]
        nfeat_level = len(feature_maps)

        # Pre-project each level to d_model when the backbone width differs.
        # No-op when feature_dim == d_model (self.feature_proj is nn.Identity).
        if not isinstance(self.feature_proj, nn.Identity):
            feature_maps = [
                self.feature_proj(fm.view(fm.shape[0] * fm.shape[1], *fm.shape[2:]))
                .view(fm.shape[0], fm.shape[1], -1, fm.shape[3], fm.shape[4])
                for fm in feature_maps
            ]

        # Shapes
        B, V, N, J, D_pose = poses_xy_norm.shape
        NJ = N * J
        B_eff = B * V  # effective batch size when flattening views

        # --- 1. Prepare per-view 2D reference points ---

        # 0..1 coords for deformable attention
        poses_xy_01 = poses_xy_norm[..., :2]  # original [0,1]
        poses_xy_01 = poses_xy_01.view(B_eff, NJ, 1, 2)

        if self.sampling_mode == "mid_level":
            if nfeat_level == 0:
                raise ValueError("mid_level sampling requires at least one feature level after the heatmap.")
            mid_level = nfeat_level // 2
            fm = feature_maps[mid_level]
            B_f, V_f, C_l, H_l, W_l = fm.shape
            assert B_f == B and V_f == V
            fm_bv = fm.view(B_eff, C_l, H_l, W_l)
            grid = torch.clamp(poses_xy_01 * 2.0 - 1.0, -1.1, 1.1)
            feats = F.grid_sample(fm_bv, grid, align_corners=False).squeeze(-1).permute(0, 2, 1)
            if query is not None:
                B_q, NJ_q, D_q = query.shape
                assert B_q == B and NJ_q == NJ, f"query.shape[1]={NJ_q} must equal N*J={NJ}"
                feats = feats + repeat(query, "B NJ D -> (B V) NJ D", V=V)
            output = self.output_proj(feats)
            C_out = output.shape[-1]
            return output.view(B, V, N, J, C_out)

        poses_xy_01_levels = poses_xy_01.expand(B_eff, NJ, nfeat_level, 2)  # (B_eff, NJ, L, 2)

        # [0, 1] -> [-1, 1] for grid_sample / deformable attention
        grid_coords = torch.clamp(poses_xy_01_levels * 2.0 - 1.0, -1.1, 1.1)

        reference_point_features = []
        for lvl in range(nfeat_level):
            fm = feature_maps[lvl]  # (B, V, C_l, H_l, W_l)
            B_f, V_f, C_l, H_l, W_l = fm.shape
            assert B_f == B and V_f == V

            # Merge (B, V) into batch
            fm_bv = fm.view(B_eff, C_l, H_l, W_l)  # (B_eff, C_l, H_l, W_l)

            # Grid for this level: (B_eff, NJ, 1, 2)
            grid_l = grid_coords[:, :, lvl : lvl + 1, :]  # (B_eff, NJ, 1, 2)

            # Sample: (B_eff, C_l, NJ, 1)
            feats = F.grid_sample(fm_bv, grid_l, align_corners=False)

            # -> (B_eff, NJ, C_l)
            feats = feats.squeeze(-1).permute(0, 2, 1)

            reference_point_features.append(feats)

        # Stack over levels: (B_eff, NJ, L, C)
        reference_point_features = torch.stack(reference_point_features, dim=2)

        if query is not None:
            B_q, NJ_q, D_q = query.shape
            assert B_q == B and NJ_q == NJ, f"query.shape[1]={NJ_q} must equal N*J={NJ}"
            # (B, NJ, D) -> (B, V, NJ, D) -> (B_eff, NJ, D)
            query_bv = repeat(query, "B NJ D -> (B V) NJ D", V=V)
            fused_q = reference_point_features + query_bv.unsqueeze(2)  # (B_eff, NJ, L, C)
        else:
            fused_q = reference_point_features  # (B_eff, NJ, L, C)

        if self.sampling_mode == "reference":
            # Ablation path: use direct per-point sampled features from the pyramid
            # without learned deformable offsets/attention weights.
            output = fused_q.mean(dim=2)
            output = self.output_proj(output)
            C_out = output.shape[-1]
            return output.view(B, V, N, J, C_out)

        # The dense flattened feature map is only needed by deformable attention.
        # Keeping this below the reference-mode return avoids materializing the
        # (B*V, sum(HW), C) tensor when sampling_mode="reference" (an ablation path;
        # the shipped assignment config uses "deformable").
        feature_maps_bv = [
            fm.view(B_eff, fm.shape[2], fm.shape[3], fm.shape[4]) for fm in feature_maps
        ]  # list of (B_eff, C_l, H_l, W_l)

        feature_maps_f = torch.cat([x.flatten(2) for x in feature_maps_bv], dim=-1).permute(
            0, 2, 1
        )  # (B_eff, S_tot, C_tot)

        value = self.value_proj(feature_maps_f)  # (B_eff, S_tot, d_model)
        value = value.view(B_eff, -1, self.n_heads, self.d_model // self.n_heads)

        # --- Predict offsets ---
        sampling_offsets_px = self.sampling_offsets(fused_q)
        # (B_eff, NJ, L, H, P, 2)
        sampling_offsets_px = sampling_offsets_px.view(B_eff, NJ, nfeat_level, self.n_heads, self.n_points, 2)
        sampling_offsets_px = sampling_offsets_px.permute(0, 1, 3, 2, 4, 5)  # (B_eff, NJ, H, L, P, 2)

        # --- Predict attention weights ---
        attention_weights = self.attention_weights(fused_q)
        attention_weights = rearrange(
            attention_weights, "b nj l (h p) -> b nj h (l p)", h=self.n_heads, p=self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = rearrange(attention_weights, "b nj h (l p) -> b nj h l p", l=nfeat_level, p=self.n_points)

        # --- Calculate sampling locations for deformable attention ---

        # poses_xy_norm_for_sampling: (B_eff, NJ, L, 2) -> (B_eff, NJ, 1, L, 1, 2)
        poses_xy_01_levels = poses_xy_01_levels.unsqueeze(2).unsqueeze(-2)

        # self.offset_normalizer: (L, 2)
        offset_norm = self.offset_normalizer.view(1, 1, 1, nfeat_level, 1, 2)

        sampling_offset = sampling_offsets_px / offset_norm  # broadcast

        sampling_locations = poses_xy_01_levels + sampling_offset  # (B_eff, NJ, H, L, P, 2)

        # --- Deformable attention ---
        output = DeformFunction.apply(
            value,
            self.spatial_shapes,  # (L, 2)
            self.feature_map_start_index,  # (L,)
            sampling_locations.contiguous(),
            attention_weights,
            B_eff,
        )

        output = self.output_proj(output)  # (B_eff, NJ, C_out)

        # --- Reshape back to (B, V, N, J, C_out) ---

        C_out = output.shape[-1]
        output = output.view(B, V, N, J, C_out)

        # Per-view attention features, shape (B, V, N, J, C_out).
        return output
