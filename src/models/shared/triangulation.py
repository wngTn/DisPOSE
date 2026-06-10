import torch
import torch.nn as nn

from src.utils.camera import camera_to_ray, get_camera_params, pixel_to_camera, world_3d_to_img_2d


class AlgebraicRayTriangulator(nn.Module):
    def __init__(self, space_configuration: dict | None = None):
        super().__init__()
        self.use_clamping = space_configuration is not None

        if self.use_clamping and space_configuration:
            # world-space box from config (in mm)
            space_size = torch.tensor(space_configuration["space_size"], dtype=torch.float32)  # (3,)
            space_center = torch.tensor(space_configuration["space_center"], dtype=torch.float32)  # (3,)

            half_extent = space_size / 2.0  # (3,)

            # store as buffers so they move with the module (cpu/gpu) and are in state_dict
            self.register_buffer("center", space_center, persistent=False)  # (3,)
            self.register_buffer("half_extent", half_extent, persistent=False)  # (3,)
        else:
            self.register_buffer("center", None, persistent=False)
            self.register_buffer("half_extent", None, persistent=False)

    def forward(
        self,
        img_xy: torch.Tensor,  # (V, *, 2) xy coordinates in (distorted) original image space
        cam_params_vec: torch.Tensor,  # (V, *, 48)
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Setup inputs
        batch_shape = img_xy.shape[1:-1]
        dtype = img_xy.dtype

        # Undistort and convert to normalized camera coordinates
        norm_cc_xy = pixel_to_camera(
            img_xy,
            cam_params_vec,
        )
        # Conver to ray in world coordinates
        d = camera_to_ray(norm_cc_xy, cam_params_vec, normalize=True)

        # Camera centers in world coords
        C = get_camera_params(cam_params_vec, ["T_c2w"])[0]  # (V, *dims, 3)

        # 2. Weights
        if weights is None:
            weights = torch.ones_like(d[..., 0])  # (V, *dims)
        weights = weights.clamp_min(0.0)
        w_scaled = weights.unsqueeze(-1)  # (V, *dims, 1)

        # 3. Skew matrix [d]_x
        zeros = torch.zeros_like(d[..., 0])
        d_skew = torch.stack(
            [zeros, -d[..., 2], d[..., 1], d[..., 2], zeros, -d[..., 0], -d[..., 1], d[..., 0], zeros],
            dim=-1,
        ).view(*d.shape[:-1], 3, 3)  # (V, *dims, 3, 3)

        # 4. Moment m = d x C
        m = torch.cross(d, C, dim=-1)  # (V, *dims, 3)

        # Build per-view 3x4 blocks [ [d]_x | -m ]
        A_block = torch.cat([d_skew, -m.unsqueeze(-1)], dim=-1)  # (V, *dims, 3, 4)

        # Apply weights
        A_block = A_block * w_scaled.unsqueeze(-1)  # (V, *dims, 3, 4)

        # 5. Stack views: (*batch_shape, V, 3, 4) -> (*batch_shape, 3V, 4)
        permute_dims = list(range(1, len(batch_shape) + 1)) + [0, len(batch_shape) + 1, len(batch_shape) + 2]
        A_stacked = A_block.permute(*permute_dims)  # (*batch_shape, V, 3, 4)
        dims_flat = A_stacked.shape[:-3]  # *batch_shape
        A_final = A_stacked.reshape(*dims_flat, -1, 4)  # (*batch_shape, 3V, 4)

        # 6. SVD solve A X' = 0
        _, S, Vh = torch.linalg.svd(A_final, full_matrices=False)  # Vh: (*batch_shape, 4, 4)
        X_hom = Vh[..., -1, :].to(dtype)  # (*batch_shape, 4)

        # TR Loss extraction: The smallest singular value is the algebraic error
        res_triang = S[..., -1].to(dtype)

        w = X_hom[..., 3:4]
        w_safe = torch.where(w.abs() < 1e-9, torch.ones_like(w), w)
        X = X_hom[..., :3] / w_safe  # (*batch_shape, 3)

        # --- Hard clamp to world bounds (if enabled) ----------------------------
        if self.use_clamping:
            min_xyz = self.center - self.half_extent
            max_xyz = self.center + self.half_extent

            X_clamped = X.clamp(min=min_xyz, max=max_xyz)  # (*batch_shape, 3)
        else:
            X_clamped = X

        # 7. Validity: require >= 2 views, a finite solution, and a non-degenerate homogeneous scale.
        tiny = torch.finfo(dtype).eps
        views_count = (weights > tiny).sum(dim=0)  # (*batch_shape)
        finite = torch.isfinite(X).all(dim=-1)  # (*batch_shape)
        good_w = (w.abs() > 1e-9).squeeze(-1)  # (*batch_shape)

        valid_mask = (views_count >= 2) & finite & good_w

        X_final = torch.where(valid_mask[..., None], X_clamped, torch.zeros_like(X_clamped))
        valid_f = valid_mask.to(dtype).unsqueeze(-1)

        return torch.cat([X_final, valid_f], dim=-1), res_triang


def triangulate_midpoint(
    img_xy: torch.Tensor,  # (V, *, 2)
    cam_params_vec: torch.Tensor,  # (V, *, 48)
    weights: torch.Tensor | None = None,  # (V, *)
    reg: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted midpoint triangulation from multiple views.

    Solves A * X = b, where X is the 3D point that minimizes the
    weighted geometric error between the point and the camera rays.

    Args:
        img_xy:         (V, *, 2) observed 2D points in (distorted) image space
        cam_params_vec: (V, *, 48) camera parameters
        weights:        (V, *) per-view weights (>= 0), defaults to ones
        reg:            Regularization for numerical stability

    Returns:
        triang_xyzs_global: (*, 4) with (x, y, z, valid_flag)
        residual:           (*,) weighted geometric error
    """
    batch_shape = img_xy.shape[1:-1]
    device = img_xy.device
    dtype = img_xy.dtype

    # Undistort and convert to normalized camera coordinates
    norm_cc_xy = pixel_to_camera(img_xy, cam_params_vec)
    d_w = camera_to_ray(norm_cc_xy, cam_params_vec, normalize=True)
    C = get_camera_params(cam_params_vec, ["T_c2w"])[0]  # (V, *dims, 3)

    # Broadcast helpers
    shape_prefix = (1,) * len(batch_shape)
    I3_brd = torch.eye(3, device=device, dtype=dtype).view(shape_prefix + (3, 3))

    # Weights
    if weights is None:
        weights = torch.ones_like(img_xy[..., 0])
    W = weights.clamp_min(0.0)

    # Build A = sum_v w_v (I - d d^T)
    W_sum = W.sum(dim=0)
    A_term1 = W_sum[..., None, None] * I3_brd

    ddT = d_w.unsqueeze(-1) * d_w.unsqueeze(-2)
    A_term2 = (W[..., None, None] * ddT).sum(dim=0)

    A = A_term1 - A_term2

    # Build b = sum_v w_v (I - d d^T) C
    b_term1 = (W[..., None] * C).sum(dim=0)
    d_dot_C = (d_w * C).sum(dim=-1)
    b_term2 = ((W * d_dot_C)[..., None] * d_w).sum(dim=0)

    bvec = b_term1 - b_term2

    # Regularize & solve A X = b
    A_reg = A + reg * I3_brd

    try:
        X_triangulated = torch.linalg.solve(A_reg, bvec[..., None]).squeeze(-1)
        finite_solve = torch.isfinite(X_triangulated).all(dim=-1)
    except torch.linalg.LinAlgError:
        X_triangulated = torch.full(bvec.shape, float("nan"), device=device, dtype=dtype)
        finite_solve = torch.zeros(batch_shape, device=device, dtype=torch.bool)

    # Compute residual (weighted sum of squared distances to rays)
    X_expanded = X_triangulated.unsqueeze(0).expand_as(C)
    V_vec = X_expanded - C
    V_dot_d = (V_vec * d_w).sum(dim=-1, keepdim=True)
    projection = V_dot_d * d_w
    perpendicular = V_vec - projection
    sq_distances = (perpendicular**2).sum(dim=-1)
    residual = (W * sq_distances).sum(dim=0)

    # Validity
    tiny = torch.finfo(dtype).eps
    views_count = (W > tiny).sum(dim=0)
    valid_triangulation = (views_count >= 2) & finite_solve

    # Format output
    X_final = torch.where(valid_triangulation[..., None], X_triangulated, torch.zeros_like(X_triangulated))
    valid_f = valid_triangulation.to(dtype).unsqueeze(-1)
    triang_xyzs_global = torch.cat([X_final, valid_f], dim=-1)

    return triang_xyzs_global, residual


def triangulate_algebraic(
    img_xy: torch.Tensor,  # (V, *, 2)
    cam_params_vec: torch.Tensor,  # (V, *, 48)
    weights: torch.Tensor,  # (V, *)
    gn_iters: int = 0,  # Gauss-Newton iterations (0 = algebraic only)
    damping: float = 1e-3,
    fd_eps: float = 1e-2,  # finite-difference step in world units
) -> torch.Tensor:
    """
    Algebraic triangulation (Plücker LS) with strict multi-view validation and
    optional Gauss-Newton refinement.

    Ensures that any point with < 2 valid views (weight > 0) returns (0,0,0).
    """
    dtype = img_xy.dtype
    device = img_xy.device

    V = img_xy.shape[0]
    mid_shape = img_xy.shape[1:-1]  # dimensions between V and the last dim (e.g. B, A, T, N)
    M = img_xy[0].numel() // 2  # total number of points per view

    # Flatten middle dimensions: (V, *, C) -> (V, M, C)
    img_xy_flat = img_xy.reshape(V, M, 2)
    cam_params_flat = cam_params_vec.reshape(V, M, -1)
    weights_flat = weights.reshape(V, M)

    # -------------------------------------------------------------------------
    # 0) Pre-calculate Validity (Requirement: >= 2 Views)
    # -------------------------------------------------------------------------
    # A point is valid for triangulation only if it is seen by at least 2 cameras.
    # We treat weight > 0 as "seen".
    views_per_point = (weights_flat > 0).sum(dim=0)  # (M,)
    is_valid_triangulation = views_per_point >= 2  # (M,)

    # -------------------------------------------------------------------------
    # 1) Algebraic triangulation (Plücker-style LS in ray space)
    # -------------------------------------------------------------------------
    # Undistort and get rays in world coordinates
    norm_cc_xy = pixel_to_camera(img_xy_flat, cam_params_flat)  # (V, M, 2)
    d = camera_to_ray(norm_cc_xy, cam_params_flat, normalize=True)  # (V, M, 3)

    # Camera centers (for Plücker)
    C = get_camera_params(cam_params_flat, ["T_c2w"])[0]  # (V, M, 3)

    # Weights
    w_scaled = weights_flat.clamp_min(0.0).unsqueeze(-1)  # (V, M, 1)

    # Skew matrices [d]_x
    zeros = torch.zeros_like(d[..., 0])
    d_skew = torch.stack(
        [
            zeros,
            -d[..., 2],
            d[..., 1],
            d[..., 2],
            zeros,
            -d[..., 0],
            -d[..., 1],
            d[..., 0],
            zeros,
        ],
        dim=-1,
    ).view(V, M, 3, 3)  # (V, M, 3, 3)

    # Moments m = d x C
    m = torch.cross(d, C, dim=-1)  # (V, M, 3)

    # Per-view 3x4 block A_i = [ [d]_x | -m ]
    # We apply weights here. If weight is 0, this view contributes 0 to the system.
    A_block = torch.cat([d_skew, -m.unsqueeze(-1)], dim=-1)  # (V, M, 3, 4)
    A_block = A_block * w_scaled.unsqueeze(-1)  # (V, M, 3, 4)

    # Stack views: (V, M, 3, 4) -> (M, V, 3, 4) -> (M, 3V, 4)
    A_stacked = A_block.permute(1, 0, 2, 3)  # (M, V, 3, 4)
    A_final = A_stacked.reshape(M, V * 3, 4)  # (M, 3V, 4)

    # SVD solve: algebraic solution in homogeneous coordinates
    # If a point has 0 or 1 view, this system is under-determined, returning
    # a least-norm solution (often near origin or on the single ray).
    _, _, Vh = torch.linalg.svd(A_final, full_matrices=False)
    X_hom = Vh[..., -1, :].to(dtype)  # (M, 4)

    # Dehomogenize
    w_h = X_hom[..., 3:4]
    w_safe = torch.where(w_h.abs() < 1e-9, torch.ones_like(w_h), w_h)
    X_flat = X_hom[..., :3] / w_safe  # (M, 3)

    # Points seen by fewer than 2 views are under-determined; force them to (0, 0, 0).
    X_flat[~is_valid_triangulation] = 0.0

    # If no GN requested, restore shape and return
    if gn_iters <= 0 or V < 2:
        X = X_flat.reshape(*mid_shape, 3)
        return X

    # -------------------------------------------------------------------------
    # 2) Gauss-Newton refinement (Finite Diff)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        img_xy_flat_det = img_xy_flat.detach()  # (V, M, 2)
        cam_params_flat_det = cam_params_flat.detach()
        weights_flat_det = weights_flat.detach()  # (V, M)

        # We reuse the validity mask calculated earlier
        valid_points = is_valid_triangulation  # (M,)

        # If absolutely no points are valid, return the algebraic result
        if not valid_points.any():
            X = X_flat.reshape(*mid_shape, 3)
            return X

        # Prepare weights for GN
        w_clamped = weights_flat_det.clamp_min(0.0)  # (V, M)
        w_sqrt = torch.where(
            w_clamped > 0.0,
            w_clamped.sqrt(),
            torch.zeros_like(w_clamped),
        )  # (V, M)

        eye3 = torch.eye(3, device=device, dtype=dtype)
        eye3_batch = eye3.unsqueeze(0).expand(M, 3, 3)  # (M, 3, 3)

        # Pre-broadcast camera params once: (V, M, 48) -> (M, V, 48)
        cam_for_reproj = cam_params_flat_det.permute(1, 0, 2).contiguous()  # (M, V, 48)

        for _ in range(gn_iters):
            # Base projection at current X_flat
            X_world = X_flat.view(M, 1, 1, 3)  # (M, 1, 1, 3)

            uv0, uv0_valid = world_3d_to_img_2d(X_world, cam_for_reproj)  # (M, V, 1, 1, 2)
            uv0 = uv0[..., 0, 0, :]  # (M, V, 2)
            uv0_valid = uv0_valid[..., 0, 0].squeeze(-1)

            proj_xy = uv0.permute(1, 0, 2)  # (V, M, 2)
            proj_xy_valid = uv0_valid.permute(1, 0, 2)

            # Residuals and weighting
            res = proj_xy - img_xy_flat_det  # (V, M, 2)
            res_w = proj_xy_valid * w_sqrt.unsqueeze(-1) * res  # (V, M, 2)

            # Finite-difference Jacobian wrt X_world (M,3)
            J_img_world = torch.empty(V, M, 2, 3, device=device, dtype=dtype)

            # Central differences along each world dimension
            for k in range(3):
                e = torch.zeros_like(X_flat)
                e[:, k] = fd_eps

                X_plus = (X_flat + e).view(M, 1, 1, 3)
                X_minus = (X_flat - e).view(M, 1, 1, 3)

                uv_plus, _ = world_3d_to_img_2d(X_plus, cam_for_reproj)
                uv_minus, _ = world_3d_to_img_2d(X_minus, cam_for_reproj)

                uv_minus = uv_minus[..., 0, 0, :]
                uv_plus = uv_plus[..., 0, 0, :]

                uv_plus = uv_plus.permute(1, 0, 2)
                uv_minus = uv_minus.permute(1, 0, 2)

                deriv_k = (uv_plus - uv_minus) / (2.0 * fd_eps)
                J_img_world[..., k] = deriv_k

            # Apply weights: J_w = w_sqrt * J
            J_w = J_img_world * w_sqrt.unsqueeze(-1).unsqueeze(-1)  # (V, M, 2, 3)

            # Normal equations per point: H (M,3,3), g (M,3)
            H = torch.einsum("vmij,vmik->mjk", J_w, J_w)  # (M, 3, 3)
            g = torch.einsum("vmij,vmi->mj", J_w, res_w)  # (M, 3)

            # Regularize Hessian (Levenberg damping)
            H_reg = H + damping * eye3_batch  # (M, 3, 3)

            # --- HANDLING INVALID POINTS ---
            # For points with < 2 views, we force delta = 0 so they stay at (0,0,0)
            invalid = ~valid_points
            if invalid.any():
                H_reg[invalid] = eye3
                g[invalid] = 0.0

            # Solve H delta = g, GN step: X_new = X - delta
            delta = torch.linalg.solve(H_reg, g.unsqueeze(-1)).squeeze(-1)  # (M, 3)
            X_flat = X_flat - delta

    # Restore original shape: (M, 3) -> (*, 3)
    X = X_flat.reshape(*mid_shape, 3)
    return X


def triangulate_final_poses_with_worst_view_dropout(
    keypoints_xys: torch.Tensor,
    cam_params_vec: torch.Tensor,
    match_map: dict[tuple[int, int], int],
    score_threshold: float = 0.3,
    min_error_threshold: float = 16.0**2,
    J: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched final 3D triangulation with worst-view dropout.

    All persons × joints are packed into (P*J, V, ...) tensors, then processed with
    two batched SVD triangulations and one batched reprojection. For each (person,
    joint) where the worst-view reprojection error exceeds ``min_error_threshold``
    (and ≥ 3 views are present), that view is dropped before the second triangulation.

    Args:
        keypoints_xys: (V, N, J, 3) — x, y, confidence score per detection.
        cam_params_vec: (V, D) or (1, V, D) camera parameter vectors.
        match_map: {(view_idx, det_idx): person_id}
        score_threshold: minimum 2D confidence to include an observation.
        min_error_threshold: squared pixel error above which the worst view is dropped.
        J: number of joints.

    Returns:
        X_final: (num_persons, J, 3) triangulated 3D coordinates (zeros where invalid).
        point_scores: (num_persons, J) mean confidence over surviving views (0.0 if invalid).
    """
    device, dtype = keypoints_xys.device, keypoints_xys.dtype
    V = keypoints_xys.shape[0]
    cam_V = cam_params_vec[0] if cam_params_vec.dim() == 3 else cam_params_vec
    assert cam_V.shape[0] == V

    if not match_map:
        return (
            keypoints_xys.new_zeros((0, J, 3)),
            torch.zeros((0, J), dtype=dtype, device=device),
        )

    num_persons = max(match_map.values()) + 1
    PJ = num_persons * J

    items = list(match_map.items())
    v_arr = torch.tensor([k[0] for k, _ in items], device=device, dtype=torch.long)
    n_arr = torch.tensor([k[1] for k, _ in items], device=device, dtype=torch.long)
    p_arr = torch.tensor([p for _, p in items], device=device, dtype=torch.long)

    M = v_arr.shape[0]
    kp = keypoints_xys[v_arr, n_arr]                              # (M, J, 3)
    valid = kp[..., 2] >= score_threshold                         # (M, J)

    j_grid = torch.arange(J, device=device).unsqueeze(0).expand(M, J)
    flat_idx = p_arr.unsqueeze(1) * J + j_grid                    # (M, J)
    v_grid = v_arr.unsqueeze(1).expand(M, J)

    obs_xy = torch.zeros((PJ, V, 2), device=device, dtype=dtype)
    obs_w = torch.zeros((PJ, V), device=device, dtype=dtype)
    fi = flat_idx[valid]
    vi = v_grid[valid]
    obs_xy[fi, vi] = kp[..., :2][valid]
    obs_w[fi, vi] = kp[..., 2][valid]

    cam_pj = cam_V.unsqueeze(0).expand(PJ, V, -1)
    n_views = (obs_w > 0).sum(dim=1)
    valid_pj = n_views >= 2

    # Transpose to (V, PJ, ...) for triangulate_algebraic
    obs_xy_t = obs_xy.permute(1, 0, 2)
    obs_w_t = obs_w.permute(1, 0)
    cam_pj_t = cam_pj.permute(1, 0, 2)
    obs_w_t_masked = obs_w_t.clone()
    obs_w_t_masked[:, ~valid_pj] = 0.0

    # --- First triangulation (algebraic SVD) ---
    X_init = triangulate_algebraic(obs_xy_t, cam_pj_t, obs_w_t_masked, gn_iters=0)

    # --- Batched reprojection for worst-view detection ---
    X_for_reproj = X_init.unsqueeze(0).unsqueeze(2)                # (1, PJ, 1, 3)
    cam_for_reproj = cam_V.unsqueeze(0)                            # (1, V, D)
    uv_reproj, _ = world_3d_to_img_2d(X_for_reproj, cam_for_reproj)
    uv_reproj = uv_reproj[0, :, :, 0, :]                           # (V, PJ, 2)

    reproj_err2 = (uv_reproj - obs_xy_t).pow(2).sum(dim=-1)        # (V, PJ)
    reproj_err2[obs_w_t <= 0] = -1.0                               # unobserved can't be worst

    # --- Worst-view dropout ---
    worst_idx = reproj_err2.argmax(dim=0)                          # (PJ,)
    worst_err = reproj_err2.gather(0, worst_idx.unsqueeze(0)).squeeze(0)

    obs_w_dropped = obs_w_t.clone()
    should_drop = (n_views > 2) & (worst_err >= min_error_threshold) & valid_pj
    if should_drop.any():
        pj_indices = torch.where(should_drop)[0]
        obs_w_dropped[worst_idx[should_drop], pj_indices] = 0.0

    # Revert if dropout left < 2 views.
    revert = (obs_w_dropped > 0).sum(dim=0) < 2
    if revert.any():
        obs_w_dropped[:, revert] = obs_w_t[:, revert]
    obs_w_dropped[:, ~valid_pj] = 0.0

    # --- Second triangulation with dropped weights ---
    X_refined = triangulate_algebraic(obs_xy_t, cam_pj_t, obs_w_dropped, gn_iters=0)
    X_final = X_refined.view(num_persons, J, 3)

    w_after = obs_w_dropped.permute(1, 0)                          # (PJ, V)
    n_kept = (w_after > 0).sum(dim=1).clamp_min(1).float()
    mean_score = w_after.sum(dim=1) / n_kept
    mean_score[~valid_pj] = 0.0
    point_scores = mean_score.view(num_persons, J)

    X_final[point_scores == 0.0] = 0.0
    return X_final, point_scores