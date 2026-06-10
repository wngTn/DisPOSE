"""Hypergraph construction for RootRegressionNet.

Encodes per-view detections into a hypergraph where each hyperedge connects
``V`` nodes (one per camera view, or a dustbin sentinel). All functions are
pure: they read submodules (``proj_attn``, ``z_cue``) as explicit arguments and
return graph tensors. The hypergraph itself has no parameters, so it does not
appear in any module's state_dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.shared.triangulation import triangulate_midpoint
from src.utils.camera import world_3d_to_img_2d


def empty_graph(device: torch.device, V: int, d_model: int) -> dict:
    """Empty graph payload with the same key set ``build_graph`` returns."""
    return {
        "X": torch.empty((0, d_model), device=device),
        "hedge_idx": torch.empty((2, 0), dtype=torch.long, device=device),
        "node_batch": torch.empty(0, dtype=torch.long, device=device),
        "edge_batch": torch.empty(0, dtype=torch.long, device=device),
        "z_cue": torch.empty(0, device=device),
        "edge_tuples": torch.empty((0, 1 + V), dtype=torch.long, device=device),
        "M": 0,
    }


def build_graph(
    feat_maps: list[torch.Tensor],
    instances: dict[str, torch.Tensor],
    cam_params: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    proj_attn: nn.Module,
    z_cue: nn.Module,
    heatmap_size: tuple[int, int],
    d_model: int,
) -> dict:
    """Build a hypergraph for cross-view matching.

    Args:
        feat_maps: backbone feature maps for the deformable-attention node read.
        instances: per-view detections (``heat_map``, ``image``, ``score``, …).
        cam_params: ``(B, V, CAM_DIM)`` camera parameters.
        node_mask: ``(B, V, N)`` bool mask of valid detections.
        proj_attn: deformable cross-view attention module.
        z_cue: geometric edge-cue module.
        heatmap_size: ``(W, H)`` of the backbone heatmap.
        d_model: feature width — used to shape the empty-graph fallback.

    Returns:
        Dict with ``X`` (node features), ``hedge_idx`` (incidence), ``node_batch``,
        ``edge_batch``, ``z_cue`` (per-edge geometric score), ``edge_tuples``, ``M``.
    """
    B, V, N = node_mask.shape
    device = node_mask.device

    # --- Node features via deformable attention ---
    node_hm = instances["heat_map"][..., :2]
    node_norm = node_hm / node_hm.new_tensor(heatmap_size)
    X = proj_attn(None, node_norm, feat_maps)[node_mask]  # (M_nodes, d)

    if X.shape[0] == 0:
        return empty_graph(device, V, d_model)

    n_counts = node_mask.view(B, -1).sum(1)
    node_batch = torch.arange(B, device=device).repeat_interleave(n_counts)

    # --- Generate candidate edges with two-level geometric filtering ---
    node_xys = instances["image"][node_mask][:, 0]  # (M_nodes, 3)
    hedge_idx, _, edge_tuples, z_cue_scores = _generate_filtered_edges(
        node_mask, node_xys, cam_params, instances["image"], z_cue=z_cue
    )

    if edge_tuples.shape[0] == 0:
        return empty_graph(device, V, d_model)

    return {
        "X": X,
        "hedge_idx": hedge_idx,
        "node_batch": node_batch,
        "edge_batch": edge_tuples[:, 0],
        "z_cue": z_cue_scores,
        "edge_tuples": edge_tuples,
        "M": edge_tuples.shape[0],
    }


@torch.no_grad()
def _build_pairwise_compat(
    node_mask: torch.Tensor,  # (B, V, N)
    inst_image: torch.Tensor,  # (B, V, N, 1, 3)
    cam_params: torch.Tensor,  # (B, V, CAM_DIM)
    *,
    z_cue: nn.Module,
) -> torch.Tensor:
    """``(B, V, V, N+1, N+1)`` bool: True when a (vi, ni) ↔ (vj, nj) pair could
    plausibly belong to an edge that survives the full geometric cue.

    Cheap 2-view triangulation test that rejects >99.9% of random combinations
    before the expensive full-edge geometric cue runs. Dustbin index ``N`` is
    always compatible.
    """
    B, V, N = node_mask.shape
    device = node_mask.device
    m_ot = N + 1

    compat = torch.ones(B, V, V, m_ot, m_ot, dtype=torch.bool, device=device)

    if not z_cue._do_threshold:
        return compat  # threshold filtering disabled → all pairs ok

    loose_thresh = z_cue.pixel_threshold * V * V

    uv = inst_image[:, :, :, 0, :2]  # (B, V, N, 2)
    wt = inst_image[:, :, :, 0, 2]  # (B, V, N)

    for vi in range(V):
        for vj in range(vi + 1, V):
            G = B * N * N

            # Expand all (ni, nj) pairs for this view pair
            uv_i = uv[:, vi].unsqueeze(2).expand(-1, -1, N, -1).reshape(G, 2)
            uv_j = uv[:, vj].unsqueeze(1).expand(-1, N, -1, -1).reshape(G, 2)
            w_i = wt[:, vi].unsqueeze(2).expand(-1, -1, N).reshape(G)
            w_j = wt[:, vj].unsqueeze(1).expand(-1, N, -1).reshape(G)

            cam_i = cam_params[:, vi, None, None, :].expand(-1, N, N, -1).reshape(G, -1)
            cam_j = cam_params[:, vj, None, None, :].expand(-1, N, N, -1).reshape(G, -1)

            # 2-view triangulation
            xyz, _ = triangulate_midpoint(
                torch.stack([uv_i, uv_j]),
                torch.stack([cam_i, cam_j]),
                torch.stack([w_i, w_j]),
            )
            xyz = xyz[:, :3]

            # Reproject + weighted reprojection error (same formula as GeometricHyperedgeCue)
            pix, pix_ok = world_3d_to_img_2d(
                xyz[:, None, None, :],
                torch.stack([cam_i, cam_j], dim=1),
            )
            pix = pix[:, :, 0, 0, :]
            pix_ok = pix_ok[:, :, 0, 0, 0]

            uv_orig = torch.stack([uv_i, uv_j], dim=1)
            sqerr = (pix - uv_orig).pow(2).sum(dim=-1)
            ww = torch.stack([w_i, w_j], dim=1).clamp_min(0.0) * pix_ok.float()
            d_geom = (ww * sqerr).sum(1) / ww.sum(1).clamp_min(1e-6)

            # Invalid pair (either detection missing) → mark compatible (dustbin
            # handles it later).
            valid_i = node_mask[:, vi].unsqueeze(2).expand(-1, -1, N).reshape(G)
            valid_j = node_mask[:, vj].unsqueeze(1).expand(-1, N, -1).reshape(G)
            pair_ok = (d_geom < loose_thresh) | ~(valid_i & valid_j)

            compat[:, vi, vj, :N, :N] = pair_ok.view(B, N, N)
            compat[:, vj, vi, :N, :N] = pair_ok.view(B, N, N).transpose(1, 2)

    return compat


@torch.no_grad()
def _generate_filtered_edges(
    node_mask: torch.Tensor,  # (B, V, N)
    node_xys: torch.Tensor,  # (M_nodes, 3)
    cam_params: torch.Tensor,  # (B, V, CAM_DIM)
    inst_image: torch.Tensor,  # (B, V, N, 1, 3)
    *,
    z_cue: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate candidate hyperedges per batch element in chunks, filter on the fly.

    1. **Pairwise pre-filter** — cheap 2-view triangulation rejects >99.9% of
       random combinations.
    2. **Full geometric cue** — applied only to the surviving set.

    Returns ``(hedge_idx, bvnj, edge_tuples, z_cue_scores)``.
    """
    B, V, N = node_mask.shape
    device = node_mask.device

    _empty = (
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty((0, 4), dtype=torch.long, device=device),
        torch.empty((0, 1 + V), dtype=torch.long, device=device),
        torch.empty(0, device=device),
    )
    if not node_mask.any():
        return _empty

    node_ranks = node_mask.reshape(-1).long().cumsum(0) - 1
    counts = node_mask.sum(dim=2)
    counts_cpu = counts.detach().cpu()
    max_opts = int(counts_cpu.max().item()) + 1
    if max_opts <= 1:
        return _empty

    # Per-(batch, view) options: [valid detection indices ..., dustbin (N)]
    options = torch.full((B, V, max_opts), N, dtype=torch.long, device=device)
    bv_pos = node_mask.long().cumsum(dim=2) - 1
    bv_b, bv_v, bv_n = node_mask.nonzero(as_tuple=True)
    options[bv_b, bv_v, bv_pos[bv_b, bv_v, bv_n]] = bv_n
    num_options = counts + 1  # (B, V)
    num_options_cpu = counts_cpu + 1

    compat = _build_pairwise_compat(node_mask, inst_image, cam_params, z_cue=z_cue)

    vi_list = [vi for vi in range(V) for _ in range(vi + 1, V)]
    vj_list = [vj for vi in range(V) for vj in range(vi + 1, V)]
    P = len(vi_list)

    CHUNK = 1 << 20  # ~1 M candidate edges per chunk

    acc_et: list[torch.Tensor] = []
    acc_zc: list[torch.Tensor] = []
    acc_ni: list[torch.Tensor] = []
    acc_ei: list[torch.Tensor] = []
    acc_bvnj: list[torch.Tensor] = []
    edge_off = 0

    for b in range(B):
        n_opts = num_options[b]
        total = int(num_options_cpu[b].prod().item())
        if total <= 1:
            continue

        strides = torch.ones(V, dtype=torch.long, device=device)
        for v in range(V - 2, -1, -1):
            strides[v] = strides[v + 1] * n_opts[v + 1]

        b_opts = options[b]
        node_base = b * V * N
        pair_compat = compat[b, vi_list, vj_list]

        for cs in range(0, total, CHUNK):
            ce = min(cs + CHUNK, total)
            C = ce - cs
            flat_idx = torch.arange(cs, ce, device=device, dtype=torch.long)

            # Decompose flat index → per-view option index
            tuples = torch.empty(C, V, dtype=torch.long, device=device)
            rem = flat_idx
            for v in range(V):
                oi = rem // strides[v]
                rem = rem - oi * strides[v]
                tuples[:, v] = b_opts[v, oi]
            del flat_idx, rem

            # Keep edges with ≥1 real (non-dustbin) detection
            keep = (tuples < N).sum(dim=1) >= 1
            if not keep.any():
                continue
            kt = tuples[keep]
            del tuples

            compatible = torch.ones(kt.shape[0], dtype=torch.bool, device=device)
            for p in range(P):
                compatible &= pair_compat[p, kt[:, vi_list[p]], kt[:, vj_list[p]]]
            if not compatible.any():
                continue
            kt = kt[compatible]
            Mc = kt.shape[0]

            et = torch.empty(Mc, 1 + V, dtype=torch.long, device=device)
            et[:, 0] = b
            et[:, 1:] = kt

            is_real = kt < N
            e_loc, v_loc = is_real.nonzero(as_tuple=True)
            n_loc = kt[e_loc, v_loc]

            gni = node_ranks[node_base + v_loc * N + n_loc]
            chunk_hi = torch.stack([gni, e_loc], dim=0)
            chunk_bvnj = torch.stack(
                [torch.full_like(v_loc, b), v_loc, n_loc, torch.zeros_like(v_loc)],
                dim=1,
            )

            # Geometric cue + threshold filter
            zc, _, hi_f, bvnj_f, kept_mask = z_cue(
                node_xys, chunk_hi, chunk_bvnj, cam_params
            )

            if zc.shape[0] == 0:
                continue

            acc_et.append(et[kept_mask])
            acc_zc.append(zc)
            hi_f[1] += edge_off
            acc_ni.append(hi_f[0])
            acc_ei.append(hi_f[1])
            acc_bvnj.append(bvnj_f)
            edge_off += zc.shape[0]

    if not acc_et:
        return _empty

    return (
        torch.stack([torch.cat(acc_ni), torch.cat(acc_ei)]),
        torch.cat(acc_bvnj),
        torch.cat(acc_et),
        torch.cat(acc_zc),
    )
