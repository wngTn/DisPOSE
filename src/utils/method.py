import torch
import torch.nn.functional as F


def nms_2d(heatmaps: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    """
    Per-heatmap 2D NMS via max-pooling.
    Input:  heatmaps of shape (*, V, C, H, W)
    Output: same shape, non-maxima suppressed.
    """
    assert kernel % 2 == 1, "kernel size should be odd"
    pad = (kernel - 1) // 2

    *leading, V, C, H, W = heatmaps.shape
    # Treat every (leading..., V, C) item as its own single-channel map
    x = heatmaps.reshape(-1, 1, H, W)  # (-1, 1, H, W)
    maxm = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=pad)
    keep = (x == maxm).to(dtype=heatmaps.dtype)
    out = x * keep
    return out.reshape(*leading, V, C, H, W)


def top_k_heatmaps(heatmaps: torch.Tensor, K: int = 10, nms_kernel: int = 5):
    """
    Top-K per heatmap (per item in (*, V, C)) on the spatial grid.

    Args:
        heatmaps: Tensor of shape (*, V, C, H, W)
        K:       number of spatial peaks to keep (clamped to H*W)
        nms_kernel: odd kernel for NMS max-pool (default 5)

    Returns:
        ind_k: (*, K, V, C, 2)  integer xy indices
        val_k: (*, K, V, C, 1)  values at those indices
    """
    hm = nms_2d(heatmaps, kernel=nms_kernel)

    *leading, V, C, H, W = hm.shape
    K_eff = min(K, H * W)

    # Flatten spatial dims; compute topk per (*, V, C)
    flat = hm.reshape(-1, H * W)  # (-1, HW)
    val_k, ind = torch.topk(flat, K_eff, dim=-1)  # (-1, K)

    # Convert flat indices to (x, y)
    x = ind % W
    y = ind // W
    ind_k = torch.stack((x, y), dim=-1)  # (-1, K, 2)

    # Reshape back to (*, V, C, K, 2)/(K, 1) then permute to (*, K, V, C, ...)
    batch_shape = (*leading, V, C)
    ind_k = ind_k.reshape(*batch_shape, K_eff, 2)
    val_k = val_k.reshape(*batch_shape, K_eff, 1)

    # Current: (*, V, C, K, 2/1) -> Desired: (*, K, V, C, 2/1)
    def _move_K_front(t: torch.Tensor):
        nd = t.dim()  # = len(leading) + 4 (V,C,K,xy/valdim)
        lead_nd = nd - 4
        # order: leading..., K, V, C, last
        perm = list(range(lead_nd)) + [lead_nd + 2, lead_nd + 0, lead_nd + 1, lead_nd + 3]
        return t.permute(*perm).contiguous()

    ind_k = _move_K_front(ind_k)
    val_k = _move_K_front(val_k)

    # If caller requested K > H*W, pad with zeros to keep the exact K shape
    if K_eff < K:
        # pad along K dimension (dim = len(leading)+0)
        pad_k = K - K_eff
        ind_pad = torch.zeros(*leading, pad_k, V, C, 2, dtype=ind_k.dtype, device=ind_k.device)
        val_pad = torch.zeros(*leading, pad_k, V, C, 1, dtype=val_k.dtype, device=val_k.device)
        ind_k = torch.cat([ind_k, ind_pad], dim=len(leading))
        val_k = torch.cat([val_k, val_pad], dim=len(leading))

    return ind_k, val_k
