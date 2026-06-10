"""Edge → 3D-pose decoding helpers for RootRegressionNet.

All functions are pure and stateless:
- ``projected_mass_to_confidence``: Sinkhorn mass × geometric cue = decode score.
- ``decode_and_triangulate``: greedy edge selection + algebraic triangulation.
- ``sample_heatmap_confidence``: optional confidence multiplier from backbone heatmaps.
- ``triangulate_gt``: GT pose triangulation by matched person IDs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.shared.triangulation import triangulate_algebraic
from src.utils.camera import world_3d_to_img_2d
from src.utils.linear_algebra import affine_transform, get_affine_transform


ROOT_JOINT_IDX = 2


@torch.no_grad()
def projected_mass_to_confidence(x_mass: torch.Tensor, z_cue: torch.Tensor) -> torch.Tensor:
    """Decode confidence = projected polytope mass × geometric cue.

    Treats the projected mass as the assignment-existence posterior on the
    feasible polytope and the geometric cue as a likelihood.
    """
    if x_mass.numel() == 0:
        return x_mass
    return x_mass.clamp_min(0.0) * z_cue.clamp_min(0.0)


@torch.no_grad()
def decode_and_triangulate(
    E_scores: torch.Tensor,           # (M,)
    edge_tuples: torch.Tensor,        # (M, 1+V)
    inst_flat: dict[str, torch.Tensor],
    cam_flat: torch.Tensor,
    heatmaps: torch.Tensor | None,    # (Bp, V, J, H, W) iff use_heatmap_confidence
    projection_params: dict[str, torch.Tensor] | None,
    *,
    num_views: int,
    matcher_threshold: float,
    use_heatmap_confidence: bool,
) -> torch.Tensor:
    """Greedy edge decode → per-slot triangulated XYZ + score.

    1. Drop edges below ``matcher_threshold``.
    2. Sort survivors by score, greedily fill ``N`` slots per batch without
       reusing nodes.
    3. Triangulate each accepted edge from its real (non-dustbin) views.
    4. If ``use_heatmap_confidence``, fold the per-view backbone heatmap value
       into the score via ``sqrt(edge_score · heatmap_conf)``.

    Returns ``(Bp, N, 4)`` with the last dim ``(x, y, z, score)``.
    """
    device = E_scores.device
    V = num_views
    Bp = cam_flat.shape[0]
    N = inst_flat["image"].shape[2]
    M = E_scores.shape[0]

    valid_mask = E_scores > matcher_threshold
    if M == 0 or not valid_mask.any():
        return torch.zeros(Bp, N, 4, device=device)

    # Sort survivors by score, descending.
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    sorted_order = torch.argsort(E_scores[valid_indices], descending=True)
    sorted_edges = valid_indices[sorted_order]

    s_batch = edge_tuples[sorted_edges, 0]
    s_nodes = edge_tuples[sorted_edges, 1:]
    s_scores = E_scores[sorted_edges]

    # Drop edges with <2 real (non-dustbin) views — can't triangulate.
    s_real = s_nodes < N
    keep = s_real.sum(dim=1) >= 2
    s_batch, s_nodes, s_scores, s_real = (
        s_batch[keep], s_nodes[keep], s_scores[keep], s_real[keep]
    )
    S = s_batch.shape[0]
    if S == 0:
        return torch.zeros(Bp, N, 4, device=device)

    # Greedy fill on CPU — sequential conflict check avoids many tiny GPU syncs.
    s_batch_c, s_nodes_c, s_real_c = s_batch.cpu(), s_nodes.cpu(), s_real.cpu()
    accepted = torch.zeros(S, dtype=torch.bool)
    slot_ids = torch.zeros(S, dtype=torch.long)
    used_nodes = torch.zeros(Bp, V, N, dtype=torch.bool)
    slots_filled = torch.zeros(Bp, dtype=torch.long)
    for i in range(S):
        b = int(s_batch_c[i])
        if slots_filled[b] >= N:
            continue
        real_mask_i = s_real_c[i]
        real_views = real_mask_i.nonzero(as_tuple=True)[0]
        real_nodes = s_nodes_c[i, real_mask_i]
        if used_nodes[b, real_views, real_nodes].any():
            continue
        used_nodes[b, real_views, real_nodes] = True
        slot_ids[i] = slots_filled[b]
        accepted[i] = True
        slots_filled[b] += 1

    acc_idx_cpu = accepted.nonzero(as_tuple=True)[0]
    if acc_idx_cpu.numel() == 0:
        return torch.zeros(Bp, N, 4, device=device)

    acc_slots = slot_ids[acc_idx_cpu].to(device)
    acc_idx = acc_idx_cpu.to(device)
    acc_b = s_batch[acc_idx]
    acc_nodes = s_nodes[acc_idx]
    acc_real = s_real[acc_idx]

    img_xy_all = inst_flat["image"]
    scores_all = inst_flat["score"]

    img_xy_vessel = torch.zeros(Bp, V, N, 1, 3, device=device)
    weights_vessel = torch.zeros(Bp, V, N, 1, device=device)
    hedge_score_vessel = torch.zeros(Bp, N, 1, device=device)

    A = acc_idx.shape[0]
    v_idx = torch.arange(V, device=device).unsqueeze(0).expand(A, -1)
    flat_b = acc_b.unsqueeze(1).expand(-1, V)[acc_real]
    flat_v = v_idx[acc_real]
    flat_slot = acc_slots.unsqueeze(1).expand(-1, V)[acc_real]
    flat_n = acc_nodes[acc_real]

    img_xy_vessel[flat_b, flat_v, flat_slot, 0] = img_xy_all[flat_b, flat_v, flat_n, 0]
    weights_vessel[flat_b, flat_v, flat_slot, 0] = scores_all[flat_b, flat_v, flat_n, 0, 0]
    hedge_score_vessel[acc_b, acc_slots, 0] = s_scores[acc_idx]

    tri_pts = rearrange(img_xy_vessel[..., :2], "b v n j c -> v b n j c")
    tri_w = rearrange(weights_vessel, "b v n j -> v b n j")
    tri_cam = repeat(cam_flat, "b v c -> v b n 1 c", n=N)
    xyz = triangulate_algebraic(tri_pts, tri_cam, tri_w)  # (Bp, N, 1, 3)

    if use_heatmap_confidence and heatmaps is not None and projection_params is not None:
        hm_conf = sample_heatmap_confidence(
            xyz[:, :, 0],
            heatmaps,
            cam_flat,
            projection_params["center"],
            projection_params["scale"],
            projection_params["rotation"],
        )
        score = torch.sqrt(hedge_score_vessel.squeeze(-1) * hm_conf).unsqueeze(-1)
    else:
        score = hedge_score_vessel.squeeze(-1).unsqueeze(-1)

    return torch.cat([xyz[:, :, 0], score], dim=-1)  # (Bp, N, 4)


@torch.no_grad()
def sample_heatmap_confidence(
    xyz: torch.Tensor,             # (Bp, N, 3)
    heatmaps: torch.Tensor,         # (Bp, V, J, H, W)
    cam_params_vec: torch.Tensor,   # (Bp, V, CAM_DIM)
    center: torch.Tensor,           # (Bp, V, 2)
    scale: torch.Tensor,            # (Bp, V, 2)
    rotation: torch.Tensor,         # (Bp, V, 1)
) -> torch.Tensor:
    """Project predicted 3D points to every view and sample the root-joint
    heatmap value. Returns ``(Bp, N)`` mean confidence across views.
    """
    Bp, N, _ = xyz.shape
    V = cam_params_vec.shape[1]
    device = xyz.device
    hm_H, hm_W = heatmaps.shape[-2:]

    xy_img, xy_valid = world_3d_to_img_2d(xyz.unsqueeze(-2), cam_params_vec)
    xy_img = xy_img.squeeze(-2)         # (Bp, V, N, 2)
    xy_valid = xy_valid.squeeze(-2).squeeze(-1)  # (Bp, V, N)

    img_wh = center * 2
    inside_w = (xy_img[..., 0] >= 0) & (xy_img[..., 0] < img_wh[:, :, None, 0])
    inside_h = (xy_img[..., 1] >= 0) & (xy_img[..., 1] < img_wh[:, :, None, 1])
    valid_proj = inside_w & inside_h & xy_valid

    hm_size_tensor = torch.tensor([hm_W, hm_H], dtype=torch.float32, device=device)
    affine = get_affine_transform(center, scale, rotation, hm_size_tensor)

    xy_img_flat = rearrange(xy_img, "b v n d -> (b v) n d")
    affine_flat = rearrange(affine, "b v ... -> (b v) ...")
    xy_hm_flat = affine_transform(xy_img_flat, affine_flat)
    xy_hm = rearrange(xy_hm_flat, "(b v) n d -> b v n d", b=Bp, v=V)

    xy_norm = (xy_hm / (hm_size_tensor.view(1, 1, 1, 2) - 1.0)) * 2.0 - 1.0
    xy_norm = xy_norm.clamp(-1.1, 1.1)

    hm_root = heatmaps[:, :, ROOT_JOINT_IDX : ROOT_JOINT_IDX + 1, :, :]
    hm_flat = rearrange(hm_root, "b v c h w -> (b v) c h w")
    grid_flat = rearrange(xy_norm, "b v n d -> (b v) 1 n d")
    sampled = F.grid_sample(hm_flat, grid_flat, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = rearrange(sampled, "(b v) 1 1 n -> b v n", b=Bp, v=V)

    per_view_conf = sampled * valid_proj.float()
    return per_view_conf.mean(dim=1)  # (Bp, N) — mean confidence over all V views


@torch.no_grad()
def triangulate_gt(
    person_ids: torch.Tensor,       # (Bp, V, N, 1, 1)
    img_xy: torch.Tensor,            # (Bp, V, N, 1, 3)
    joint_scores: torch.Tensor,      # (Bp, V, N, 1, 1)
    cam_params_vec: torch.Tensor,    # (Bp, V, CAM_DIM)
    *,
    num_views: int,
) -> torch.Tensor:
    """Triangulate GT poses by grouping matched person IDs across views."""
    V = num_views
    Bp = cam_params_vec.shape[0]
    N = img_xy.shape[2]
    device = img_xy.device

    gt_img_xy = torch.zeros(Bp, V, N, 1, 3, device=device)
    gt_w = torch.zeros(Bp, V, N, 1, device=device)

    pid = person_ids.squeeze(-1).squeeze(-1)        # (Bp, V, N)
    scores = joint_scores.squeeze(-1).squeeze(-1)    # (Bp, V, N)

    for b in range(Bp):
        valid = (pid[b] != -1) & (scores[b] > 0)
        if not valid.any():
            continue
        uniq = torch.unique(pid[b][valid], sorted=True)[:N]
        P = uniq.numel()
        if P == 0:
            continue

        flat_pid = pid[b].reshape(-1)
        flat_valid = valid.reshape(-1)
        slots = torch.searchsorted(uniq, flat_pid)
        slots.clamp_max_(P - 1)
        matched = flat_valid & (uniq[slots] == flat_pid) & (slots < N)
        if not matched.any():
            continue

        idx = matched.nonzero(as_tuple=True)[0]
        v_idx = idx // N
        n_idx = idx % N
        s_idx = slots[idx]

        gt_img_xy[b, v_idx, s_idx, 0] = img_xy[b, v_idx, n_idx, 0]
        gt_w[b, v_idx, s_idx, 0] = scores[b, v_idx, n_idx]

    tri_pts = rearrange(gt_img_xy[..., :2], "b v n j c -> v b n j c")
    tri_w = rearrange(gt_w, "b v n j -> v b n j")
    tri_cam = repeat(cam_params_vec, "b v c -> v b n 1 c", n=N)
    xyz = triangulate_algebraic(tri_pts, tri_cam, tri_w)
    return torch.cat([xyz, gt_w.mean(1).unsqueeze(-1)], dim=-1)
