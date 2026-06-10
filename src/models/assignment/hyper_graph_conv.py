import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_max

from timm.layers.mlp import Mlp


def init_linear_xavier(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MPHyperGraphConv(MessagePassing):
    """
    DiT-style symmetric hypergraph layer:
      - Node -> Edge attention (+ competitive max feature)
      - Edge FFN
      - Edge -> Node attention (+ competitive max feature)
      - Node FFN

    Conditioning enters via AdaLN parameters produced from:
      - node_cond: (N, D)  (typically time embedding broadcast to nodes)
      - edge_cond: (M, D)  (typically time + z cue embedding broadcast to edges)

    Design notes:
      1) Max-pool feature uses the *value-projected* space (consistent with attention values).
      2) scatter_max is sanitized (no -inf poisoning when receivers have zero incidences).
      3) Output projections accept concatenated [attn, max] features (2*D -> D).
      4) AdaLN final linear is zero-init so gates start at 0 (identity start).
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads
        self.dropout_p = dropout

        # -------------------------
        # Node -> Edge projections
        # -------------------------
        self.W_q_edge = nn.Linear(d_model, d_model, bias=False)
        self.W_k_node = nn.Linear(d_model, d_model, bias=False)
        self.W_v_node = nn.Linear(d_model, d_model, bias=False)
        self.W_o_edge = nn.Linear(d_model * 2, d_model)  # [attn, max] -> D

        # -------------------------
        # Edge -> Node projections
        # -------------------------
        self.W_q_node = nn.Linear(d_model, d_model, bias=False)
        self.W_k_edge = nn.Linear(d_model, d_model, bias=False)
        self.W_v_edge = nn.Linear(d_model, d_model, bias=False)
        self.W_o_node = nn.Linear(d_model * 2, d_model)  # [attn, max] -> D

        # -------------------------
        # FFNs
        # -------------------------
        self.ffn_node = Mlp(d_model, d_model * 4, d_model, act_layer=nn.SiLU, drop=dropout)  # type: ignore
        self.ffn_edge = Mlp(d_model, d_model * 4, d_model, act_layer=nn.SiLU, drop=dropout)  # type: ignore

        # -------------------------
        # AdaLN (8 blocks of D):
        # [g_src,b_src,g_tgt,b_tgt, gate_attn, g_ffn,b_ffn, gate_ffn]
        # -------------------------
        self.adaLN_node = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 8 * d_model))
        self.adaLN_edge = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 8 * d_model))

        # Base init
        self.apply(init_linear_xavier)

        # Zero-init final AdaLN linear so gates start at 0 (identity start)
        with torch.no_grad():
            nn.init.zeros_(self.adaLN_node[-1].weight)
            nn.init.zeros_(self.adaLN_node[-1].bias)
            nn.init.zeros_(self.adaLN_edge[-1].weight)
            nn.init.zeros_(self.adaLN_edge[-1].bias)

    @staticmethod
    def _sanitize_max(x: torch.Tensor) -> torch.Tensor:
        # scatter_max returns -inf for receivers with no messages.
        # Replace non-finite entries with zeros to avoid poisoning residuals.
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    def forward(
        self,
        x_nodes: torch.Tensor,  # (N, D)
        x_edges: torch.Tensor,  # (M, D)
        edge_index: torch.Tensor,  # (2, E) [node_idx, hedge_idx]
        node_cond: torch.Tensor,  # (N, D)
        edge_cond: torch.Tensor,  # (M, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = x_nodes.size(0)
        M = x_edges.size(0)

        # -------------------------
        # AdaLN params for nodes
        # -------------------------
        c_n = self.adaLN_node(node_cond).chunk(8, dim=-1)
        n_g_src, n_b_src = c_n[0], c_n[1]
        n_g_tgt, n_b_tgt = c_n[2], c_n[3]
        n_gate_attn = c_n[4]
        n_g_ffn, n_b_ffn = c_n[5], c_n[6]
        n_gate_ffn = c_n[7]

        # -------------------------
        # AdaLN params for edges
        # -------------------------
        c_e = self.adaLN_edge(edge_cond).chunk(8, dim=-1)
        e_g_tgt, e_b_tgt = c_e[0], c_e[1]
        e_g_src, e_b_src = c_e[2], c_e[3]
        e_gate_attn = c_e[4]
        e_g_ffn, e_b_ffn = c_e[5], c_e[6]
        e_gate_ffn = c_e[7]

        # ============================================================
        # Phase 1: Nodes -> Edges
        # ============================================================
        # Target: edges (queries)
        x_e_q = F.layer_norm(x_edges, (self.d_model,)) * (1.0 + e_g_tgt) + e_b_tgt
        # Source: nodes (keys/values)
        x_n_kv = F.layer_norm(x_nodes, (self.d_model,)) * (1.0 + n_g_src) + n_b_src

        q_edge = self.W_q_edge(x_e_q).view(M, self.heads, self.d_head)  # (M, H, Dh)
        k_node = self.W_k_node(x_n_kv).view(N, self.heads, self.d_head)  # (N, H, Dh)

        # Value projection in full D space (for consistent max feature)
        v_node_full = self.W_v_node(x_n_kv)  # (N, D)
        v_node = v_node_full.view(N, self.heads, self.d_head)  # (N, H, Dh)

        # Attention aggregated message: (M, H, Dh) -> (M, D)
        out_attn_e = self.propagate(edge_index, q=q_edge, k=k_node, v=v_node, size=(N, M))
        out_attn_e = out_attn_e.reshape(M, self.d_model)

        # Competitive max message in the SAME value-projected space: (M, D)
        src_for_max_e = v_node_full[edge_index[0]]  # (E, D)
        out_max_e, _ = scatter_max(src_for_max_e, edge_index[1], dim=0, dim_size=M)
        out_max_e = self._sanitize_max(out_max_e)

        out_fused_e = torch.cat([out_attn_e, out_max_e], dim=-1)  # (M, 2D)
        x_edges = x_edges + e_gate_attn * self.W_o_edge(out_fused_e)

        # Edge FFN
        x_e_norm = F.layer_norm(x_edges, (self.d_model,)) * (1.0 + e_g_ffn) + e_b_ffn
        x_edges = x_edges + e_gate_ffn * self.ffn_edge(x_e_norm)

        # ============================================================
        # Phase 2: Edges -> Nodes
        # ============================================================
        # Target: nodes (queries)
        x_n_q = F.layer_norm(x_nodes, (self.d_model,)) * (1.0 + n_g_tgt) + n_b_tgt
        # Source: edges (keys/values)
        x_e_kv = F.layer_norm(x_edges, (self.d_model,)) * (1.0 + e_g_src) + e_b_src

        q_node = self.W_q_node(x_n_q).view(N, self.heads, self.d_head)  # (N, H, Dh)
        k_edge = self.W_k_edge(x_e_kv).view(M, self.heads, self.d_head)  # (M, H, Dh)

        v_edge_full = self.W_v_edge(x_e_kv)  # (M, D)
        v_edge = v_edge_full.view(M, self.heads, self.d_head)  # (M, H, Dh)

        rev_edge_index = edge_index.flip(0)  # [hedge_idx, node_idx]

        out_attn_n = self.propagate(rev_edge_index, q=q_node, k=k_edge, v=v_edge, size=(M, N))
        out_attn_n = out_attn_n.reshape(N, self.d_model)

        src_for_max_n = v_edge_full[rev_edge_index[0]]  # (E, D)
        out_max_n, _ = scatter_max(src_for_max_n, rev_edge_index[1], dim=0, dim_size=N)
        out_max_n = self._sanitize_max(out_max_n)

        out_fused_n = torch.cat([out_attn_n, out_max_n], dim=-1)  # (N, 2D)
        x_nodes = x_nodes + n_gate_attn * self.W_o_node(out_fused_n)

        # Node FFN
        x_n_norm = F.layer_norm(x_nodes, (self.d_model,)) * (1.0 + n_g_ffn) + n_b_ffn
        x_nodes = x_nodes + n_gate_ffn * self.ffn_node(x_n_norm)

        return x_nodes, x_edges

    def message(self, q_i, k_j, v_j, index, ptr, size_i): # type: ignore
        """
        q_i: (E, H, Dh) from target nodes (receiver side)
        k_j: (E, H, Dh) from source nodes (sender side)
        v_j: (E, H, Dh) from source nodes (sender side)
        index: receiver indices (used for softmax normalization)
        """
        attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.d_head)  # (E, H)
        attn = softmax(attn, index, ptr, size_i)  # normalize over senders per receiver
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        return v_j * attn.unsqueeze(-1)  # (E, H, Dh)
