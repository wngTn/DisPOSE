"""
**Convention: Column Vectors**

All 3D transformations use a column-vector convention.
The core world-to-camera transformation is:
$x_c = R_{w2c} @ x_w + t_{w2c}$

Where:
- $x_w$: 3D point in world coordinates (a $3 \times 1$ vector).
- $x_c$: 3D point in camera coordinates (a $3 \times 1$ vector).
- $R_{w2c}$: The $3 \times 3$ world-to-camera rotation matrix.
- $t_{w2c}$: The $3 \times 1$ world-to-camera translation vector.
"""

import torch
from einops import rearrange

# Define the parameter map at the module level.
# This map stores:
# key: (slice_object, target_shape_for_non_batch_dims)
# The slice is for the last dimension (-1) of the 51-element vector.
CAM_PARAM_MAP: dict[str, tuple[slice, tuple[int, ...]]] = {
    # 9 elements, reshapes to (3, 3)
    "R_w2c": (slice(0, 9), (3, 3)),
    # 9 elements, reshapes to (3, 3)
    "R_c2w": (slice(9, 18), (3, 3)),
    # 3 elements, final shape (*, 3)
    "T_w2c": (slice(18, 21), (3,)),
    # 3 elements, final shape (*, 3)
    "T_c2w": (slice(21, 24), (3,)),
    # 9 elements, reshapes to (3, 3)
    "K": (slice(24, 33), (3, 3)),
    # 9 elements, reshapes to (3, 3)
    "K_inv": (slice(33, 42), (3, 3)),
    # 6 elements (Rational Model), final shape (*, 6)
    "k": (slice(42, 48), (6,)),
    # 2 elements, final shape (*, 2)
    "p": (slice(48, 50), (2,)),
    # 1 element, final shape (*, 1)
    "cam_idx": (slice(50, 51), (1,)),
}
"""
Total elements: 9+9+3+3+9+9+6+2+1 = 51
"""


def get_camera_params(
    cam_vector: torch.Tensor,
    keys: list[str],
) -> list[torch.Tensor]:
    """
    Extracts and reshapes specified parameters from a flattened camera tensor.

    The input vector must be a PyTorch tensor and can have
    any number of batch dimensions, e.g., (51,), (B, 51), (B, T, 51), etc.

    Args:
        cam_vector: A PyTorch tensor of shape (*, 51).
        keys: A list of strings corresponding to the parameters to extract.
              e.g., ["R_w2c", "T_w2c"].

    Returns:
        A list of the requested parameters, reshaped to their proper
        dimensions (e.g., (*, 3, 3) or (*, 3)).

    Raises:
        ValueError: If the last dimension of cam_vector is not 51.
        KeyError: If a key in `keys` is not valid.
        TypeError: If the input is not a torch.Tensor.
    """

    # --- Input Validation ---
    if cam_vector.shape[-1] != 51:
        raise ValueError(f"Input vector's last dimension must be 51, but got shape {cam_vector.shape}")

    output_params = []
    for key in keys:
        if key not in CAM_PARAM_MAP:
            raise KeyError(f"Unknown camera parameter key: '{key}'. Valid keys are: {list(CAM_PARAM_MAP.keys())}")

        slice_obj, target_shape = CAM_PARAM_MAP[key]
        sliced_param = cam_vector[..., slice_obj]
        if len(target_shape) == 2:  # This is a 3x3 matrix (e.g., R_w2c, K)
            h, w = target_shape
            # '... (h w)' -> '... h w'
            reshaped_param = rearrange(sliced_param, "... (h w) -> ... h w", h=h, w=w)
        else:  # This is a vector (e.g., T_w2c, k, p)
            # No reshaping is needed.
            reshaped_param = sliced_param
        output_params.append(reshaped_param)
    return output_params


def world_3d_to_img_2d(
    X: torch.Tensor,  # (..., N, J, 3)
    cam_params_vec: torch.Tensor,  # (..., V, 51)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D world points X into pixel coordinates using the Rational Distortion Model.
    Includes dimension-agnostic safety clamps to prevent NaN/Inf during training.
    """
    # --- Broadcast camera params to match X ---
    if cam_params_vec.ndim < X.ndim - 1:
        need = (X.ndim - 1) - cam_params_vec.ndim
        cam_params_vec = cam_params_vec.view((1,) * need + cam_params_vec.shape)

    R_w2c, t_w2c, K, k, p = get_camera_params(cam_params_vec, ["R_w2c", "T_w2c", "K", "k", "p"])

    # ---- World -> Camera
    X_cam = torch.einsum("...vcd,...njd->...njvc", R_w2c, X)
    X_cam = X_cam + t_w2c[..., None, None, :, :]

    # ---- Normalize with z guard
    z = X_cam[..., 2:3]
    valid_z = z > 1e-6
    z = z.clamp_min(1e-6)

    normalized = X_cam[..., :2] / z
    normalized = torch.where(valid_z, normalized, torch.zeros_like(normalized))

    x_n = normalized[..., 0]
    y_n = normalized[..., 1]

    # ---- Radial Calculation
    r2 = x_n * x_n + y_n * y_n

    # r2=25.0 means radius=5.0 (approx 78 degrees off-axis).
    # Anything beyond this is mathematically unstable for rational models anyway.
    r2 = r2.clamp(max=25.0)

    r4 = r2 * r2
    r6 = r4 * r2

    # Per-view coefficients -> insert N,J dims
    # k has shape (..., V, 6)
    k1 = k[..., 0][..., None, None, :]
    k2 = k[..., 1][..., None, None, :]
    k3 = k[..., 2][..., None, None, :]
    k4 = k[..., 3][..., None, None, :]
    k5 = k[..., 4][..., None, None, :]
    k6 = k[..., 5][..., None, None, :]

    p1 = p[..., 0][..., None, None, :]
    p2 = p[..., 1][..., None, None, :]

    # Rational Model
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6

    # Prevent division by zero, but preserving gradient sign direction
    denominator = torch.where(torch.abs(denominator) < 1e-6, torch.sign(denominator + 1e-9) * 1e-6, denominator)

    radial = numerator / denominator

    # A distortion factor of 20x or -20x is physically nonsensical.
    # Clamping this prevents the optimizer from seeing gradients of 1e5.
    radial = torch.clamp(radial, min=-20.0, max=20.0)

    # Tangential Distortion
    tan_x = 2.0 * p1 * x_n * y_n + p2 * (r2 + 2.0 * x_n * x_n)
    tan_y = p1 * (r2 + 2.0 * y_n * y_n) + 2.0 * p2 * x_n * y_n

    xd = x_n * radial + tan_x
    yd = y_n * radial + tan_y

    distorted_norm = torch.stack([xd, yd], dim=-1)

    # ---- Intrinsics -> pixels
    f = torch.stack([K[..., 0, 0], K[..., 1, 1]], dim=-1)[..., None, None, :, :]
    c = K[..., :2, 2][..., None, None, :, :]

    pix = distorted_norm * f + c
    pix = rearrange(pix, "... n j v c -> ... v n j c")
    valid_mask = rearrange(valid_z, "... n j v c -> ... v n j c")

    return pix, valid_mask


def undistort(
    pixels_ij: torch.Tensor,  # (*, 2) distorted pixels
    cam_params_vec: torch.Tensor,  # (*, 51) broadcastable
    num_iters: int = 5,
) -> torch.Tensor:
    """
    Undistort pixels and return shape (*, 2).
    Handles Rational Distortion Model (k1-k6, p1-p2).

    Args:
        pixels_ij: (*, 2) distorted pixel coordinates
        cam_params_vec: (*, 51) camera parameters, broadcastable with pixels_ij
        num_iters: number of iterative refinement steps

    Returns:
        (*, 2) undistorted pixel coordinates
    """
    K, k, p = get_camera_params(cam_params_vec, ["K", "k", "p"])
    # K: (*, 3, 3), k: (*, 6), p: (*, 2)

    # Build OpenCV style vector for rational model (length 8)
    # Order: [k1, k2, p1, p2, k3, k4, k5, k6]
    dist8 = torch.stack(
        [
            k[..., 0],  # k1
            k[..., 1],  # k2
            p[..., 0],  # p1
            p[..., 1],  # p2
            k[..., 2],  # k3
            k[..., 3],  # k4
            k[..., 4],  # k5
            k[..., 5],  # k6
        ],
        dim=-1,
    )

    # Guard tiny focal lengths
    eps = 1e-8 if torch.finfo(K.dtype).eps < 1e-12 else 1e-8
    K_safe = K.clone()
    K_safe[..., 0, 0] = K_safe[..., 0, 0].clamp_min(eps)
    K_safe[..., 1, 1] = K_safe[..., 1, 1].clamp_min(eps)

    # Run undistortion
    undistorted = undistortPoints_torch(
        points=pixels_ij,  # (*, 2)
        K=K_safe,  # (*, 3, 3)
        dist=dist8,  # (*, 8)
        new_K=K_safe,  # (*, 3, 3)
        num_iters=num_iters,
    )

    # Clean numerical issues
    if torch.isnan(undistorted).any() or torch.isinf(undistorted).any():
        undistorted = torch.nan_to_num(undistorted, neginf=0.0, posinf=0.0)

    return undistorted


def pixel_to_camera(
    pixels_ij: torch.Tensor,  # (*, 2) distorted pixels
    cam_params_vec: torch.Tensor,  # (*, 51) broadcastable
    num_iters: int = 5,
) -> torch.Tensor:
    """
    Distorted pixels -> normalized camera coordinates.

    Steps:
      1) Undistort (u_d, v_d) -> (u, v)
      2) Homogenize to (u, v, 1)
      3) Apply K^{-1}
      4) Dehomogenize

    Args:
        pixels_ij: (*, 2) distorted pixel coordinates
        cam_params_vec: (*, 51) camera parameters, broadcastable with pixels_ij
        num_iters: undistortion iterations

    Returns:
        (*, 2) normalized camera coordinates
    """
    # 1) Undistort: (*, 2)
    uv_und = undistort(pixels_ij, cam_params_vec, num_iters=num_iters)

    # 2) Homogeneous pixels: (*, 3)
    ones = torch.ones_like(uv_und[..., :1])
    uv1 = torch.cat([uv_und, ones], dim=-1)

    # 3) Apply K^{-1}
    (K_inv,) = get_camera_params(cam_params_vec, ["K_inv"])  # (*, 3, 3)

    # x_cam_h = K_inv @ [u, v, 1]^T
    # einsum: (*, 3, 3) @ (*, 3) -> (*, 3)
    xh = torch.einsum("...ab,...b->...a", K_inv, uv1)

    # 4) Dehomogenize
    z = xh[..., 2:3].clamp_min(torch.finfo(xh.dtype).eps)
    xy = xh[..., :2] / z  # (*, 2)

    return xy


def camera_to_ray(
    xy_norm: torch.Tensor,  # (*, 2) normalized camera coords
    cam_params_vec: torch.Tensor,  # (*, 51) broadcastable
    normalize: bool = False,
) -> torch.Tensor:
    """
    Normalized camera coords -> world-frame ray directions.

    Column-vector convention:
      - Camera-frame direction: d_c = [x, y, 1]^T
      - World-frame direction:  d_w = R_c2w @ d_c

    Args:
        xy_norm: (*, 2) normalized camera coordinates
        cam_params_vec: (*, 51) camera parameters, broadcastable
        normalize: if True, L2-normalize the output rays

    Returns:
        (*, 3) ray directions in world coordinates
    """
    # Camera-frame directions [x, y, 1]: (*, 3)
    ones = torch.ones_like(xy_norm[..., :1])
    d_cam = torch.cat([xy_norm, ones], dim=-1)

    # Rotate to world frame
    (R_c2w,) = get_camera_params(cam_params_vec, ["R_c2w"])  # (*, 3, 3)

    # d_w = R_c2w @ d_c
    # einsum: (*, 3, 3) @ (*, 3) -> (*, 3)
    rays_w = torch.einsum("...ab,...b->...a", R_c2w, d_cam)

    if normalize:
        eps = torch.finfo(rays_w.dtype).eps
        rays_w = rays_w / torch.linalg.norm(rays_w, dim=-1, keepdim=True).clamp_min(eps)

    return rays_w


def undistortPoints_torch(points, K, dist, new_K=None, num_iters=5):
    """
    Undistort points using iterative method.

    Args:
        points: (*, 2) distorted pixel coordinates
        K: (*, 3, 3) camera intrinsics, broadcastable with points
        dist: (*, D) distortion coefficients where D in [4, 5, 8, 12, 14]
        new_K: (*, 3, 3) new intrinsics for output, defaults to K
        num_iters: number of iterations

    Returns:
        (*, 2) undistorted pixel coordinates
    """
    if points.shape[-1] != 2:
        raise ValueError(f"points shape is invalid. Got {points.shape}.")
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")
    if new_K is None:
        new_K = K
    elif new_K.shape[-2:] != (3, 3):
        raise ValueError(f"new_K matrix shape is invalid. Got {new_K.shape}.")
    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}")

    # --- Basic sanitization on inputs ---
    points = torch.nan_to_num(points, neginf=0.0, posinf=0.0)

    # Adding zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])

    # Clamp distortion coefficients
    dist = torch.clamp(dist, -1e3, 1e3)

    # Extract intrinsics as (*, ) tensors for proper broadcasting with (*, 2) points
    cx = K[..., 0, 2]  # (*, )
    cy = K[..., 1, 2]
    fx = K[..., 0, 0].clamp_min(1e-8)
    fy = K[..., 1, 1].clamp_min(1e-8)

    # Convert 2D points from pixels to normalized camera coordinates
    x = (points[..., 0] - cx) / fx  # (*, )
    y = (points[..., 1] - cy) / fy

    # Initial clamp of normalized coords
    max_norm_xy = 1e4
    x = x.clamp(-max_norm_xy, max_norm_xy)
    y = y.clamp(-max_norm_xy, max_norm_xy)

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        inv_tilt = tilt_projection(dist[..., 12], dist[..., 13], True)
        xy_stack = torch.stack([x, y], dim=-1)  # (*, 2)
        x, y = transform_points(inv_tilt, xy_stack).unbind(-1)

    x0, y0 = x, y

    max_r2 = 1e4
    max_inv_rad = 1e4

    # Extract distortion coefficients as (*, ) for broadcasting
    k1 = dist[..., 0]
    k2 = dist[..., 1]
    p1 = dist[..., 2]
    p2 = dist[..., 3]
    k3 = dist[..., 4]
    k4 = dist[..., 5]
    k5 = dist[..., 6]
    k6 = dist[..., 7]
    s1 = dist[..., 8]
    s2 = dist[..., 9]
    s3 = dist[..., 10]
    s4 = dist[..., 11]

    for _ in range(num_iters):
        r2 = (x * x + y * y).clamp_max(max_r2)

        # Numerator and denominator of radial polynomial
        num = 1 + k4 * r2 + k5 * r2 * r2 + k6 * r2**3
        den = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2**3
        den = den.clamp_min(1e-6)
        inv_rad_poly = (num / den).clamp(-max_inv_rad, max_inv_rad)

        deltaX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2 * r2
        deltaY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r2 * r2

        x = (x0 - deltaX) * inv_rad_poly
        y = (y0 - deltaY) * inv_rad_poly

        x = x.clamp(-max_norm_xy, max_norm_xy)
        y = y.clamp(-max_norm_xy, max_norm_xy)

    # Convert back to pixel coordinates
    new_cx = new_K[..., 0, 2]
    new_cy = new_K[..., 1, 2]
    new_fx = new_K[..., 0, 0].clamp_min(1e-8)
    new_fy = new_K[..., 1, 1].clamp_min(1e-8)

    x = new_fx * x + new_cx
    y = new_fy * y + new_cy

    out = torch.stack([x, y], dim=-1)  # (*, 2)
    out = torch.nan_to_num(out, neginf=0.0, posinf=0.0)

    return out


def tilt_projection(taux, tauy, return_inverse=False):
    r"""Estimate the tilt projection matrix or the inverse tilt projection matrix.
    Args:
        taux: Rotation angle in radians around the :math:`x`-axis with shape :math:`(*, 1)`.
        tauy: Rotation angle in radians around the :math:`y`-axis with shape :math:`(*, 1)`.
        return_inverse: False to obtain the the tilt projection matrix. True for the inverse matrix.
    Returns:
        torch.Tensor: Inverse tilt projection matrix with shape :math:`(*, 3, 3)`.
    """
    if taux.shape != tauy.shape:
        raise ValueError(f"Shape of taux {taux.shape} and tauy {tauy.shape} do not match.")

    ndim: int = taux.dim()
    taux = taux.reshape(-1)
    tauy = tauy.reshape(-1)

    cTx = torch.cos(taux)
    sTx = torch.sin(taux)
    cTy = torch.cos(tauy)
    sTy = torch.sin(tauy)
    zero = torch.zeros_like(cTx)
    one = torch.ones_like(cTx)

    Rx = torch.stack([one, zero, zero, zero, cTx, sTx, zero, -sTx, cTx], -1).reshape(-1, 3, 3)
    Ry = torch.stack([cTy, zero, -sTy, zero, one, zero, sTy, zero, cTy], -1).reshape(-1, 3, 3)
    R = Ry @ Rx

    if return_inverse:
        invR22 = 1 / R[..., 2, 2]
        invPz = torch.stack(
            [invR22, zero, R[..., 0, 2] * invR22, zero, invR22, R[..., 1, 2] * invR22, zero, zero, one], -1
        ).reshape(-1, 3, 3)

        inv_tilt = R.transpose(-1, -2) @ invPz
        if ndim == 0:
            inv_tilt = torch.squeeze(inv_tilt)

        return inv_tilt

    Pz = torch.stack(
        [R[..., 2, 2], zero, -R[..., 0, 2], zero, R[..., 2, 2], -R[..., 1, 2], zero, zero, one], -1
    ).reshape(-1, 3, 3)

    tilt = Pz @ R.transpose(-1, -2)
    if ndim == 0:
        tilt = torch.squeeze(tilt)

    return tilt


def convert_points_to_homogeneous(points):
    r"""Function that converts points from Euclidean to homogeneous space.
    Args:
        points: the points to be transformed with shape :math:`(B, N, D)`.
    Returns:
        the points in homogeneous coordinates :math:`(B, N, D+1)`.
    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def convert_points_from_homogeneous(points, eps: float = 1e-8):
    r"""Function that converts points from homogeneous to Euclidean space.
    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.
    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.
    Examples:
        >>> input = torch.tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # we check for points at max_val
    z_vec = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def transform_points(trans_01, points_1):
    r"""Function that applies transformations to a set of points.
    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.
    Shape:
        - Output: :math:`(B, N, D)`
    Examples:
        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """

    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError(
            f"Input batch size must be the same for both tensors or 1.Got {trans_01.shape} and {points_1.shape}"
        )
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError(f"Last input dimensions must differ by one unitGot{trans_01} and {points_1}")

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0
