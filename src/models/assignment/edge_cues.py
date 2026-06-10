import torch
import torch.nn as nn

from src.models.shared.triangulation import triangulate_midpoint
from src.utils.camera import world_3d_to_img_2d

CAM_PARAMS_DIM = 51


def _small_int_tensor_to_cpu(t: torch.Tensor) -> torch.Tensor:
    """Materialize tiny integer metadata tensors on CPU for Python control flow."""
    return t.detach().to(device="cpu")


class GeometricHyperedgeCue(nn.Module):
    """
    Computes geometric cues for hyperedges based on reprojection error.

    For multi-view edges (>=2 views):
      1) Triangulate a 3D point from all involved views.
      2) Compute reprojection error.
      3) Convert to exponential cue: exp(-λ * error / ν).
      4) Optionally filter by pixel threshold.

    For singleton edges (1 view):
      - Assign a fixed prior value (no geometric information available).

    Args:
        num_views: Number of camera views.
        lam: Lambda for exponential cue computation.
        pixel_threshold: Squared pixel error threshold for filtering. Set to -1 to disable.
        singleton_prior: Prior cue value for singleton edges.
    """

    def __init__(
        self,
        num_views: int,
        lam: float = 0.01,
        pixel_threshold: float = 64**2,
        singleton_prior: float = 0.3,
    ):
        super().__init__()

        assert num_views >= 2, f"Need at least 2 views, got {num_views}"
        self.num_views = num_views
        self.lambda_reproj = lam
        self.pixel_threshold = pixel_threshold
        self.singleton_prior = singleton_prior

        popcount_lut = torch.tensor(
            [bin(x).count("1") for x in range(1 << num_views)],
            dtype=torch.long,
        )
        self.register_buffer("popcount_lut", popcount_lut, persistent=False)

    @property
    def _do_threshold(self) -> bool:
        return self.pixel_threshold >= 0

    def forward(
        self,
        node_img_xys: torch.Tensor,  # (M_nodes, 3) - [u, v, weight]
        global_hedge_inc: torch.Tensor,  # (2, E_inc) - [node_idx, edge_idx]
        bvnj_lookup: torch.Tensor,  # (E_inc, 4) - [batch, view, detection, joint]
        cam_params_vec: torch.Tensor,  # (B, V, CAM_DIM)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute geometric cues for all edges.

        Returns:
            edge_cues: (H_kept,) cue value per kept edge
            inc_cues: (E_inc_kept,) cue value per kept incidence
            global_hedge_inc: (2, E_inc_kept) filtered and reindexed incidence matrix
            bvnj_lookup: (E_inc_kept, 4) filtered lookup
            kept_edge_mask: (H_orig,) boolean mask of kept edges (before reindexing)
        """
        device, dtype = node_img_xys.device, node_img_xys.dtype
        E_inc = global_hedge_inc.shape[1]

        if E_inc == 0:
            return self._empty_result(device, dtype)

        V = cam_params_vec.shape[1]
        assert V == self.num_views

        inc_nodes = global_hedge_inc[0].long()
        inc_edges = global_hedge_inc[1].long()
        b_idx = bvnj_lookup[:, 0].long()
        v_idx = bvnj_lookup[:, 1].long()
        j_idx = bvnj_lookup[:, 3].long()

        H = int(_small_int_tensor_to_cpu(inc_edges).max().item()) + 1

        # --- Count incidences per edge ---
        edge_counts = torch.bincount(inc_edges, minlength=H)
        is_singleton = edge_counts == 1
        is_multiview = edge_counts >= 2

        # --- Initialize output cues ---
        edge_cues = torch.full((H,), self.singleton_prior, device=device, dtype=dtype)
        inc_cues = torch.full((E_inc,), self.singleton_prior, device=device, dtype=dtype)
        keep_edges = torch.ones(H, device=device, dtype=torch.bool)  # Keep all by default

        # --- Process multi-view edges ---
        if is_multiview.any():
            mv_edge_cues, mv_inc_cues, mv_keep_mask = self._compute_multiview_cues(
                node_img_xys,
                inc_nodes,
                inc_edges,
                b_idx,
                v_idx,
                j_idx,
                cam_params_vec,
                is_multiview,
                H,
                E_inc,
            )

            # Update cues for multi-view edges
            edge_cues[is_multiview] = mv_edge_cues

            # Update incidence cues for multi-view edges
            mv_inc_mask = is_multiview[inc_edges]
            inc_cues[mv_inc_mask] = mv_inc_cues[mv_inc_mask]

            # Apply threshold filtering if enabled
            if self._do_threshold:
                keep_edges = is_singleton.clone()  # Always keep singletons
                keep_edges[is_multiview] = mv_keep_mask

        # --- Filter and reindex outputs ---
        return self._filter_outputs(edge_cues, inc_cues, global_hedge_inc, bvnj_lookup, keep_edges)

    def _compute_multiview_cues(
        self,
        node_img_xys: torch.Tensor,
        inc_nodes: torch.Tensor,
        inc_edges: torch.Tensor,
        b_idx: torch.Tensor,
        v_idx: torch.Tensor,
        j_idx: torch.Tensor,
        cam_params_vec: torch.Tensor,
        is_multiview: torch.Tensor,
        H: int,
        E_inc: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute geometric cues for multi-view edges.

        Returns:
            mv_edge_cues: (H_mv,) cues for multi-view edges only
            inc_cues: (E_inc,) cues for all incidences (singleton = prior)
            keep_mask: (H_mv,) boolean mask for threshold filtering
        """
        device, dtype = node_img_xys.device, node_img_xys.dtype
        V = cam_params_vec.shape[1]

        # Filter to multi-view incidences
        mv_inc_mask = is_multiview[inc_edges]
        mv_inc_indices = mv_inc_mask.nonzero(as_tuple=True)[0]

        mv_nodes = inc_nodes[mv_inc_indices]
        mv_edges = inc_edges[mv_inc_indices]
        mv_b = b_idx[mv_inc_indices]
        mv_v = v_idx[mv_inc_indices]
        mv_j = j_idx[mv_inc_indices]

        # --- Group by (batch, edge, joint) ---
        keys = torch.stack([mv_b, mv_edges, mv_j], dim=1)
        unique_keys, group_indices = torch.unique(keys, dim=0, return_inverse=True)
        G = unique_keys.shape[0]

        # --- Compute slot indices using view masks ---
        pow2_v = (1 << mv_v).long()
        group_view_mask = torch.zeros(G, dtype=torch.long, device=device)
        group_view_mask.index_add_(0, group_indices, pow2_v)

        prefix_pop = torch.zeros(G, V, dtype=torch.long, device=device)
        for v in range(V):
            prefix_pop[:, v] = self.popcount_lut[group_view_mask & ((1 << v) - 1)]
        m_idx = prefix_pop[group_indices, mv_v]

        # --- Build padded tensors ---
        M_MAX = self.num_views
        padded_weights = torch.zeros(G, M_MAX, device=device, dtype=dtype)
        padded_cams = torch.zeros(G, M_MAX, CAM_PARAMS_DIM, device=device, dtype=dtype)
        padded_uv = torch.zeros(G, M_MAX, 2, device=device, dtype=dtype)
        mask = torch.zeros(G, M_MAX, device=device, dtype=torch.bool)

        padded_weights[group_indices, m_idx] = node_img_xys[mv_nodes, 2]
        padded_cams[group_indices, m_idx] = cam_params_vec[mv_b, mv_v]
        padded_uv[group_indices, m_idx] = node_img_xys[mv_nodes, :2]
        mask[group_indices, m_idx] = True

        # --- Triangulate ---
        Xw, _ = triangulate_midpoint(
            padded_uv.transpose(0, 1),
            padded_cams.transpose(0, 1),
            padded_weights.transpose(0, 1),
        )
        Xw = Xw[..., :3]

        # --- Reprojection error ---
        weights = padded_weights.clamp_min(0.0) * mask
        d_view, d_geom = self._compute_reprojection_error(Xw, padded_cams, padded_uv, weights, mask)

        # --- Compute cues ---
        m_eff = (weights > 0).sum(dim=1)
        nu = ((m_eff - 1) ** (self.num_views - 1)).clamp_min(1)

        c_group = torch.exp(-self.lambda_reproj * d_geom / nu)
        c_view = torch.exp(-self.lambda_reproj * d_view / nu.unsqueeze(-1))
        c_view = torch.where(mask, c_view, torch.zeros_like(c_view))

        # --- Map back to edges and incidences ---
        edge_ids = unique_keys[:, 1]

        # Average cue per edge (over joints)
        mv_edge_cues_full = torch.zeros(H, device=device, dtype=dtype)
        cnt = torch.zeros(H, device=device, dtype=dtype)
        mv_edge_cues_full.index_add_(0, edge_ids, c_group)
        cnt.index_add_(0, edge_ids, torch.ones_like(c_group))
        mv_edge_cues_full = mv_edge_cues_full / cnt.clamp_min(1.0)

        # Per-incidence cues
        inc_cues = torch.full((E_inc,), self.singleton_prior, device=device, dtype=dtype)
        inc_cues[mv_inc_indices] = c_view[group_indices, m_idx]

        # --- Threshold mask ---
        if self._do_threshold:
            edge_err_sum = torch.zeros(H, device=device, dtype=dtype)
            edge_err_cnt = torch.zeros(H, device=device, dtype=dtype)
            edge_err_sum.index_add_(0, edge_ids, d_geom)
            edge_err_cnt.index_add_(0, edge_ids, torch.ones_like(d_geom))
            avg_err = edge_err_sum / edge_err_cnt.clamp_min(1.0)
            keep_mask_full = avg_err <= self.pixel_threshold
        else:
            keep_mask_full = torch.ones(H, device=device, dtype=torch.bool)

        # Extract only multi-view results
        mv_edge_cues = mv_edge_cues_full[is_multiview]
        keep_mask = keep_mask_full[is_multiview]

        return mv_edge_cues, inc_cues, keep_mask

    def _compute_reprojection_error(
        self,
        X: torch.Tensor,  # (G, 3)
        cams: torch.Tensor,  # (G, M, CAM_DIM)
        uv_meas: torch.Tensor,  # (G, M, 2)
        weights: torch.Tensor,  # (G, M)
        mask: torch.Tensor,  # (G, M)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reprojection errors."""
        G = X.shape[0]
        uv_pred, uv_valid = world_3d_to_img_2d(X.view(G, 1, 1, 3), cams)
        uv_pred = uv_pred.view(G, -1, 2)
        uv_valid = uv_valid.view(G, -1)

        sqerr = (uv_pred - uv_meas).pow(2).sum(dim=2)
        w = weights.clamp_min(0.0) * mask * uv_valid
        d_view = w * sqerr
        d_geom = d_view.sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)

        return d_view, d_geom

    def _filter_outputs(
        self,
        edge_cues: torch.Tensor,  # (H,)
        inc_cues: torch.Tensor,  # (E_inc,)
        hedge_inc: torch.Tensor,  # (2, E_inc)
        bvnj: torch.Tensor,  # (E_inc, 4)
        keep_edges: torch.Tensor,  # (H,) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filter outputs to kept edges and reindex.

        Returns kept_edge_mask BEFORE reindexing so caller can use it.
        """
        device = edge_cues.device
        H = edge_cues.shape[0]
        H_kept = int(_small_int_tensor_to_cpu(keep_edges).sum().item())

        if H_kept == 0:
            return self._empty_result(device, edge_cues.dtype)

        # Return early if nothing filtered
        if H_kept == H:
            return edge_cues, inc_cues, hedge_inc, bvnj, keep_edges

        # Remap edge indices
        new_edge_idx = torch.full((H,), -1, device=device, dtype=torch.long)
        new_edge_idx[keep_edges] = torch.arange(H_kept, device=device)

        # Filter incidences
        inc_edges = hedge_inc[1].long()
        keep_inc = keep_edges[inc_edges]

        hedge_inc_new = hedge_inc[:, keep_inc].clone()
        hedge_inc_new[1] = new_edge_idx[hedge_inc_new[1].long()]

        return (
            edge_cues[keep_edges],
            inc_cues[keep_inc],
            hedge_inc_new,
            bvnj[keep_inc],
            keep_edges,  # Return original mask for caller
        )

    def _empty_result(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(0, device=device, dtype=dtype),
            torch.zeros(0, device=device, dtype=dtype),
            torch.zeros((2, 0), device=device, dtype=torch.long),
            torch.zeros((0, 4), device=device, dtype=torch.long),
            torch.zeros(0, device=device, dtype=torch.bool),
        )
