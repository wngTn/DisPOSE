import torch


def affine_transform(pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Apply per-view 2x3 affine transforms to points.

    Args:
        pts: (V, *, 2)  — last dim are (x, y), leading dim is view V
        t:   (V, 2, 3)  — per-view affine matrices (row-vector convention)

    Returns:
        out: (V, *, 2)
    """
    if pts.ndim < 2 or pts.shape[-1] != 2:
        raise ValueError(f"`pts` must have last dim=2, got shape {pts.shape}")
    if t.ndim != 3 or t.shape[1:] != (2, 3):
        raise ValueError(f"`t` must be (V, 2, 3), got shape {t.shape}")
    if pts.shape[0] != t.shape[0]:
        raise ValueError(f"Leading dim (views) must match: pts V={pts.shape[0]} vs t V={t.shape[0]}")

    V = pts.shape[0]
    device = pts.device
    dtype = pts.dtype

    # Flatten arbitrary middle dims to a single dimension for matmul
    pts_flat = pts.reshape(V, -1, 2)  # (V, M, 2)
    ones = torch.ones(V, pts_flat.shape[1], 1, device=device, dtype=dtype)
    pts_h = torch.cat([pts_flat, ones], dim=-1)  # (V, M, 3)

    # Multiply by per-view affine: (V, M, 3) @ (V, 3, 2) -> (V, M, 2)
    out_flat = pts_h @ t.transpose(1, 2)

    # Restore original middle dims
    out = out_flat.reshape(pts.shape[:-1] + (2,))
    return out


def transform_to_original_img_space(coords, center, scale, rot, output_size, use_udp=False):
    """
    Args:
        coords (torch.Tensor[*, K, ndims]): Predicted keypoint coordinates.
            ndims can be 2 for (x, y), or more to include scores and tags.
        center (torch.Tensor[*, 2]): Bounding box centers (x, y).
        scale (torch.Tensor[*, 2]): Bounding box scales wrt [width, height].
        rot (torch.Tensor[*]): Rotation angles in degrees.
        output_size (list[2] | torch.Tensor[2]): Size of the destination
            heatmaps (shared across all inputs).
        use_udp (bool): Use unbiased data processing. Defaults to False.

    Returns:
        torch.Tensor[*, K, ndims]: Predicted coordinates in the image space.
    """
    # --- Input Validation and Preparation ---
    # More general assertions that check relative shapes
    prefix_shape = coords.shape[:-2]
    assert coords.shape[-1] >= 2, "coords must have at least 2 dimensions for (x, y)"
    assert center.shape == prefix_shape + (2,), "Shape of center must be (*, 2)"
    assert scale.shape == prefix_shape + (2,), "Shape of scale must be (*, 2)"
    assert rot.shape == prefix_shape, "Shape of rot must be (*)"
    assert len(output_size) == 2

    # Ensure all inputs are torch tensors on the correct device
    center = torch.as_tensor(center, dtype=torch.float32, device=coords.device)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=coords.device)
    rot = torch.as_tensor(rot, dtype=torch.float32, device=coords.device)
    output_size = torch.as_tensor(output_size, dtype=torch.float32, device=coords.device)

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    # --- Transformation Calculation ---

    # Calculate the scaling factor. `unsqueeze(-2)` adds a dimension of size 1
    # before the last dimension, making its shape [*, 1, 2].
    # This ensures it broadcasts correctly with the keypoints dimension K.
    if use_udp:
        scale_factor = (scale / (output_size - 1.0)).unsqueeze(-2)
    else:
        scale_factor = (scale / output_size).unsqueeze(-2)

    # Use ellipsis (...) to slice all dimensions up to the last one.
    xy_coords = coords[..., 0:2]

    # Step 1: Center coordinates by subtracting the heatmap center.
    # heatmap_center is [2,], and PyTorch broadcasts it automatically
    # for the subtraction from xy_coords [*, K, 2].
    heatmap_center = output_size * 0.5
    xy_coords = xy_coords - heatmap_center

    # Step 2: Scale the coordinates.
    # Broadcasting [*, K, 2] * [*, 1, 2] -> [*, K, 2]
    xy_coords = xy_coords * scale_factor

    # Step 3: Rotate the coordinates
    rad = torch.deg2rad(rot)
    cos_r = torch.cos(rad)
    sin_r = torch.sin(rad)

    # Create rotation matrices of shape [*, 2, 2]
    # We build the last two dimensions and let the prefix dimensions `*` remain.
    row1 = torch.stack([cos_r, -sin_r], dim=-1)  # shape [*, 2]
    row2 = torch.stack([sin_r, cos_r], dim=-1)  # shape [*, 2]
    rot_mat = torch.stack([row1, row2], dim=-2)  # shape [*, 2, 2]

    # Apply the rotation. We transpose the inner 2x2 matrices.
    # torch.matmul handles the broadcasting over all prefix dimensions `*`.
    # It performs the operation on [*, K, 2] and [*, 2, 2], resulting in [*, K, 2].
    xy_coords = torch.matmul(xy_coords, rot_mat.transpose(-2, -1))

    # Step 4: Translate coordinates to the image's bounding box center
    # Unsqueeze center to [*, 1, 2] for broadcasting over the K dimension.
    xy_coords = xy_coords + center.unsqueeze(-2)

    # --- Finalize Output ---
    target_coords = coords.clone()
    # Use ellipsis slicing again for assignment
    target_coords[..., 0:2] = xy_coords

    return target_coords


def get_affine_transform(
    center: torch.Tensor,  # (*, 2) or (2,)
    scale: torch.Tensor,  # (*, 2) or (2,)
    rot: torch.Tensor,  # (*, 1) or (1,)
    output_size,  # (2,) or (*, 2) or tuple/list
    shift=(0.0, 0.0),  # (2,) or (*, 2) or tuple/list
    inv: bool = False,
) -> torch.Tensor:
    """
    Batched 2x3 affine transform (Torch-only, robust broadcasting).
    Returns: (*, 2, 3) — and if no batch was provided, returns (2, 3).
    """
    if not isinstance(center, torch.Tensor) or not isinstance(scale, torch.Tensor) or not isinstance(rot, torch.Tensor):
        raise TypeError("center, scale, rot must be torch.Tensors")

    dtype = center.dtype
    device = center.device

    # --- remember if user passed scalars-without-batch ---
    no_batch_input = center.ndim == 1 and scale.ndim == 1 and rot.ndim == 1

    center = center.to(dtype=dtype, device=device)
    scale = scale.to(dtype=dtype, device=device)
    rot = rot.to(dtype=dtype, device=device)

    output_size = _to_tensor_like(output_size, center)
    shift = _to_tensor_like(shift, center)
    no_batch_input = no_batch_input and (output_size.ndim == 1) and (shift.ndim == 1)

    # Ensure trailing dims and at least one batch dim
    center = _ensure_batch_lastdim(center, 2)
    scale = _ensure_batch_lastdim(scale, 2)
    rot = _ensure_batch_lastdim(rot, 1)
    output_size = _ensure_batch_lastdim(output_size, 2)
    shift = _ensure_batch_lastdim(shift, 2)

    assert center.shape[-1] == 2 and scale.shape[-1] == 2 and rot.shape[-1] == 1
    assert output_size.shape[-1] == 2 and shift.shape[-1] == 2

    # Broadcast & expand
    batch_shape = torch.broadcast_shapes(
        center.shape[:-1], scale.shape[:-1], rot.shape[:-1], output_size.shape[:-1], shift.shape[:-1]
    )
    def bexpand(x):
        return x.expand(batch_shape + x.shape[-1:])

    center, scale, rot, output_size, shift = map(bexpand, (center, scale, rot, output_size, shift))

    # Build src/dst keypoints
    scale_tmp = scale * 200.0
    src_w, src_h = scale_tmp[..., 0], scale_tmp[..., 1]
    dst_w, dst_h = output_size[..., 0], output_size[..., 1]

    rot_rad = torch.pi * rot[..., 0] / 180.0
    cos_rot, sin_rot = torch.cos(rot_rad), torch.sin(rot_rad)

    condition = src_w >= src_h
    src_dir_x_unrot = torch.where(condition, torch.zeros_like(src_w), -0.5 * src_h)
    src_dir_y_unrot = torch.where(condition, -0.5 * src_w, torch.zeros_like(src_h))
    src_dir = _get_dir(src_dir_x_unrot, src_dir_y_unrot, cos_rot, sin_rot)

    dst_dir_x = torch.where(condition, torch.zeros_like(dst_w), -0.5 * dst_h)
    dst_dir_y = torch.where(condition, -0.5 * dst_w, torch.zeros_like(dst_h))
    dst_dir = _stack2(dst_dir_x, dst_dir_y)

    true_center = center + scale_tmp * shift
    dst_center = output_size * 0.5

    src_0, src_1 = true_center, true_center + src_dir
    src_2 = _get_3rd_point(src_0, src_1)
    dst_0, dst_1 = dst_center, dst_center + dst_dir
    dst_2 = _get_3rd_point(dst_0, dst_1)

    src = torch.stack((src_0, src_1, src_2), dim=-2)  # (*, 3, 2)
    dst = torch.stack((dst_0, dst_1, dst_2), dim=-2)  # (*, 3, 2)

    p_src, p_dst = (dst, src) if inv else (src, dst)

    ones = torch.ones(p_src.shape[:-1] + (1,), dtype=dtype, device=device)  # (*, 3, 1)
    S = torch.cat((p_src, ones), dim=-1)  # (*, 3, 3)

    M_T = torch.linalg.pinv(S) @ p_dst  # (*, 3, 2)
    M = M_T.transpose(-1, -2)  # (*, 2, 3)

    # --- if user passed no batch, return (2,3) instead of (1,2,3) ---
    if no_batch_input:
        M = M.squeeze(0)

    return M


# --- Utility ---


def _to_tensor_like(x, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x, (tuple, list)):
        return torch.tensor(x, dtype=ref.dtype, device=ref.device)
    if isinstance(x, (int, float)):
        return torch.tensor([x], dtype=ref.dtype, device=ref.device)
    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected Tensor/tuple/list/number.")
    return x.to(dtype=ref.dtype, device=ref.device)


def _ensure_batch_lastdim(x: torch.Tensor, lastdim: int) -> torch.Tensor:
    """
    Ensure x has shape (..., lastdim) with an explicit batch dim (>=1).
    If x is (lastdim,), make it (1, lastdim).
    """
    if x.ndim == 1 and x.shape[0] == lastdim:
        return x.unsqueeze(0)  # (1, lastdim)
    if x.ndim == 0 and lastdim == 1:
        return x.view(1, 1)  # scalar -> (1,1) when needed
    return x


def _stack2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack((a, b), dim=-1)


def _get_3rd_point(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    direct = a - b
    ortho_vec = torch.stack((-direct[..., 1], direct[..., 0]), dim=-1)
    return b + ortho_vec


def _get_dir(x: torch.Tensor, y: torch.Tensor, cos_rot: torch.Tensor, sin_rot: torch.Tensor) -> torch.Tensor:
    x_out = x * cos_rot - y * sin_rot
    y_out = x * sin_rot + y * cos_rot
    return _stack2(x_out, y_out)
