import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.mlp import Mlp
from torch_scatter import scatter_add

from src.models.assignment.hyper_graph_conv import MPHyperGraphConv


def init_linear_small(m: nn.Linear, std: float = 1e-3) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class GCNDenoiser(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        node_in_dim: int = 256,
        edge_in_dim: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        noise_gate_static_conditioning: bool = False,
        noise_gate_static_floor: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.noise_gate_static_conditioning = bool(noise_gate_static_conditioning)
        # When the gate is active, the static-cue scale at noise level α̅_t is
        #   floor + (1 − floor) · sqrt(1 − α̅_t)
        # floor=0.0 silences static cues completely at the clean endpoint (t→0);
        # floor=0.5 keeps them at half strength there. Higher noise always passes
        # static cues at full strength.
        self.noise_gate_static_floor = float(noise_gate_static_floor)

        # 1. Embedders
        self.t_sincos = SinCosTimestepEmbedding(d_model)
        self.t_mlp = Mlp(d_model, d_model, d_model, act_layer=nn.SiLU)  # type: ignore
        self.z_embedder = ZCueEmbedder(d_model=d_model, hidden=d_model)

        # 2. Projections
        self.node_proj = nn.Linear(node_in_dim, d_model)
        self.edge_proj = nn.Linear(edge_in_dim, d_model)

        # 3. Conditioning Blocks
        # Z-Cue FiLM: Multiplicative conditioning for edge reliability
        self.z_film = FeatureFiLM(d_model)

        # 4. Backbone
        self.layers = nn.ModuleList(
            [MPHyperGraphConv(d_model, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm_edges = nn.LayerNorm(d_model)

        self.edge_attention = GraphAttention(d_model, num_heads=heads, dropout=dropout)
        self.head = HyperEdgeHead(d_model=d_model, d_hidden=d_model, drop=dropout, use_sum_pooling=False)

    def forward(
        self,
        X_v: torch.Tensor,  # (N_total, node_in_dim)
        W_t: torch.Tensor,  # (M_total,) diffusion state feature
        z_cue: torch.Tensor,  # (M_total,)
        t_batch: torch.Tensor,  # (B,) integer timestep t cast to float
        global_hedge_idx: torch.Tensor,  # (2, E_total)
        node_batch_idx: torch.Tensor,  # (N_total,)
        edge_batch_idx: torch.Tensor,  # (M_total,)
        noise_level: torch.Tensor | None = None,  # (B,) sqrt(1 - α̅_t) — required when noise_gate_static_conditioning=True
    ) -> torch.Tensor:
        if W_t.numel() == 0:
            return W_t.new_zeros((0,))

        # ---------------------------
        # 1. Global Time Embedding
        # ---------------------------
        t_emb = self.t_mlp(self.t_sincos(t_batch))  # (B, D)
        if self.noise_gate_static_conditioning:
            if noise_level is None:
                raise ValueError(
                    "noise_gate_static_conditioning=True requires `noise_level` "
                    "(sqrt(1-α̅_t)) to be passed in. The caller "
                    "(ProjectedConstrainedDDIM.predict_u0_from_xt) computes this from "
                    "the schedule buffer."
                )
            floor = self.noise_gate_static_floor
            # static cues at full strength when noise is high (sqrt(1-α̅_t)→1) and
            # silenced toward `floor` when noise is low (sqrt(1-α̅_t)→0).
            static_scale_batch = (
                floor + (1.0 - floor) * noise_level.to(t_emb.dtype)
            ).unsqueeze(-1)
        else:
            static_scale_batch = torch.ones((t_batch.shape[0], 1), device=t_batch.device, dtype=t_emb.dtype)

        # Distribute time to nodes and edges (Broadcast)
        node_time = t_emb[node_batch_idx]  # (N, D)
        edge_time = t_emb[edge_batch_idx]  # (M, D)
        node_static_scale = static_scale_batch[node_batch_idx]
        edge_static_scale = static_scale_batch[edge_batch_idx]

        # ---------------------------
        # 2. Local Z-Cue Embedding
        # ---------------------------
        z_emb = self.z_embedder(z_cue) * edge_static_scale  # (M, D)

        # ---------------------------
        # 3. Input Projection + Injection
        # ---------------------------
        if W_t.dim() == 1:
            state_features = W_t.unsqueeze(-1)
        elif W_t.dim() == 2:
            state_features = W_t
        else:
            raise ValueError(f"Expected diffusion state features with shape (M,) or (M,C), got {tuple(W_t.shape)}")

        h_W_t_embedding = self.edge_proj(state_features)

        # Initial features
        h_edges = h_W_t_embedding.clone()
        h_nodes = self.node_proj(X_v) * node_static_scale

        # Add Time (Global Shift)
        h_nodes = h_nodes + node_time
        h_edges = h_edges + edge_time

        # Multiplicative Z-Cue
        h_edges = self.z_film(h_edges, z_emb)

        # ---------------------------
        # 4. Hypergraph Backbone
        # ---------------------------
        edge_cond_combined = edge_time + z_emb

        for layer in self.layers:
            h_edges = h_edges + h_W_t_embedding
            h_nodes, h_edges = layer(
                h_nodes, h_edges, global_hedge_idx, node_cond=node_time, edge_cond=edge_cond_combined
            )

        # ---------------------------
        # 5. Refinement & Head
        # ---------------------------
        h_edges = self.final_norm_edges(h_edges)
        h_edges = self.edge_attention(h_edges, edge_batch_idx)

        u0_hat = self.head(h_nodes, h_edges, global_hedge_idx)
        return u0_hat


class SinCosTimestepEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for scalar diffusion noise conditions.
    Produces (B, d_model).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) int/long or float
        """
        dim = self.d_model
        half_dim = dim // 2
        device = t.device

        # Avoid half_dim-1 = 0 edge case
        denom = max(half_dim - 1, 1)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=device) / denom)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*half_dim)

        # If dim is odd, pad one
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb


class ZCueEmbedder(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None, eps: float = 1e-6):
        super().__init__()
        hidden = hidden or d_model
        self.eps = eps
        self.mlp = Mlp(1, hidden, d_model, act_layer=nn.SiLU)  # type: ignore

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (M,) or (M,1) in [0,1]
        returns: (M, d_model)
        """
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        z = z.clamp(self.eps, 1.0 - self.eps)
        z_logit = torch.log(z) - torch.log1p(-z)  # logit
        return self.mlp(z_logit)


class FeatureFiLM(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model * 2)

        # Zero-init: start as identity mapping (gamma=0, beta=0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (M, d_model)
        cond_emb: (M, d_model) derived from z_cue
        """
        # Predict scale and shift from conditioning
        scale_shift = self.proj(cond_emb)
        gamma, beta = scale_shift.chunk(2, dim=-1)

        # Modulate
        # We norm x before modulation to ensure gamma/beta act on a stable distribution
        x_norm = self.norm(x)
        return x_norm * (1.0 + gamma) + beta


class HyperEdgeHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        drop: float = 0.0,
        use_sum_pooling: bool = False,
    ):
        super().__init__()
        self.use_sum_pooling = use_sum_pooling

        self.member_mlp = Mlp(d_model, d_hidden, d_hidden, act_layer=nn.SiLU, drop=drop)  # type: ignore
        self.set_mlp = Mlp(d_hidden, d_hidden, d_hidden, act_layer=nn.SiLU, drop=drop)  # type: ignore

        self.edge_to_film_trunk = nn.Sequential(nn.Linear(d_model, d_hidden * 2), nn.SiLU())
        self.edge_to_film_head = nn.Linear(d_hidden * 2, d_hidden * 2)

        self.classifier = nn.Linear(d_hidden + d_model, 1)

        # Init
        # FiLM head zero => gamma=0, beta=0 initially (identity modulation)
        nn.init.zeros_(self.edge_to_film_head.weight)
        nn.init.zeros_(self.edge_to_film_head.bias)

        # Small-init classifier => near-zero outputs
        init_linear_small(self.classifier, std=1e-3)

    def forward(
        self,
        h_nodes: torch.Tensor,  # (N, d_model)
        h_edges: torch.Tensor,  # (M, d_model)
        hyper_edge_indices: torch.Tensor,  # (2, E) with [node_idx, hedge_idx]
    ) -> torch.Tensor:
        node_idx, hedge_idx = hyper_edge_indices
        M = h_edges.size(0)
        device = h_edges.device

        # Transform member nodes
        node_h_trans = self.member_mlp(h_nodes)  # (N, d_hidden)
        members_h = node_h_trans[node_idx]  # (E, d_hidden)

        # Pool to edges (count-safe)
        if self.use_sum_pooling:
            set_sum = scatter_add(members_h, hedge_idx, dim=0, dim_size=M)
            set_agg = set_sum
        else:
            set_sum = scatter_add(members_h, hedge_idx, dim=0, dim_size=M)
            counts = scatter_add(
                torch.ones((members_h.size(0),), device=device, dtype=set_sum.dtype),
                hedge_idx,
                dim=0,
                dim_size=M,
            ).clamp_min(1.0)
            set_agg = set_sum / counts.unsqueeze(-1)

        # FiLM modulation from edge embedding
        film_feat = self.edge_to_film_trunk(h_edges)  # (M, 2*d_hidden)
        gamma_beta = self.edge_to_film_head(film_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated = (1.0 + gamma) * set_agg + beta

        # Post-process
        set_repr = self.set_mlp(modulated)  # (M, d_hidden)
        combined = torch.cat([set_repr, h_edges], dim=-1)  # (M, d_hidden + d_model)

        # Predict Logits
        logits = self.classifier(combined).squeeze(-1)  # (M,)
        return logits


class GraphAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.ff = Mlp(d_model, d_model * 4, d_model, act_layer=nn.SiLU, drop=dropout)  # type: ignore
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        x: (M_total, D)
        batch_idx: (M_total,)
        """
        if x.size(0) == 0:
            return x

        device = x.device
        batch_size = int(batch_idx.max().item()) + 1
        counts = torch.bincount(batch_idx, minlength=batch_size)
        max_len = int(counts.max().item())

        if max_len == 0:
            return x

        # Vectorized densification: compute per-element position within its batch group
        order = batch_idx.argsort(stable=True)
        running = torch.arange(order.numel(), device=device)
        group_starts = counts.cumsum(0) - counts  # start index of each group in sorted order
        within_group_pos = running - group_starts[batch_idx[order]]
        group_pos = torch.zeros_like(batch_idx)
        group_pos[order] = within_group_pos

        dense = torch.zeros((batch_size, max_len, x.size(-1)), device=device, dtype=x.dtype)
        mask = torch.ones((batch_size, max_len), device=device, dtype=torch.bool)  # True=pad

        dense[batch_idx, group_pos] = x
        mask[batch_idx, group_pos] = False

        all_padded = mask.all(dim=1)
        if all_padded.any():
            # Unmask one dummy token to avoid NaNs
            mask[all_padded, 0] = False

        attn_out, _ = self.attn(dense, dense, dense, key_padding_mask=mask)

        if all_padded.any():
            attn_out[all_padded] = 0.0

        y = self.norm1(dense + self.drop(attn_out))
        ff_out = self.ff(y)

        if all_padded.any():
            ff_out[all_padded] = 0.0

        y = self.norm2(y + ff_out)

        # Vectorized reconstruction: gather from dense back to flat
        out = y[batch_idx, group_pos]

        return out
