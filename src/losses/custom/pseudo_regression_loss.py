import math
from itertools import combinations

import torch
import torch.nn as nn

from src.utils.camera import world_3d_to_img_2d


class _OneSidedL2(torch.autograd.Function):
    """Memory-efficient backward for ``Σ w · relu(target − pred)² / norm``.

    The naive autograd graph for ``relu(t-p).pow(2).mul(w).sum() / norm`` saves four
    full per-element tensors for backward (``t-p``, ``relu(t-p)``, ``(...)²``,
    ``...·w``). At the per-person heatmap shape (B, A, T, V, N, J, H, W) each of
    those is ~1 GiB at bs=4, so the chain's working set is ~4 GiB per stage.

    This Function instead saves only ``diff = relu(t-p)`` and ``weights``. The
    gradient w.r.t. pred is ``-2 · diff · weights / norm``; ``target`` is GT and
    has no gradient anyway. Saves ~3 of the 4 chained intermediates per call.
    """

    @staticmethod
    def forward(ctx, pred_hm, target_hm, weights, norm):
        diff = torch.clamp(target_hm - pred_hm, min=0.0)
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
        # diff² · weights_expanded summed → scalar; intermediates fall out of scope
        loss = (diff * diff * weights_expanded).sum() / norm
        ctx.save_for_backward(diff, weights)
        ctx.norm = norm
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        diff, weights = ctx.saved_tensors
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
        # d loss / d pred = -2 · diff · w / norm  (relu mask is built into diff itself)
        grad_pred = (-2.0 / ctx.norm) * diff * weights_expanded * grad_output
        return grad_pred, None, None, None


def original_image_resolution(image_size) -> tuple[int, int]:
    """Original camera resolution ``(width, height)`` for the dataset behind a given
    network input size. MM-OR uses 2048x1536 Azure-Kinect color frames; the other
    datasets use 1920x1080. Used to bound reprojected 3D joints to the image frame.
    """
    img_width = 2048 if image_size[0] == 768 else 1920
    img_height = 1536 if image_size[1] == 576 else 1080
    return img_width, img_height


def process_pseudo_2d(
    gt_poses_xyzs: torch.Tensor,  # (*, N, J, 4) (x,y,z,score3d)
    gt_poses_xys: torch.Tensor,  # (*, V, N, J, 3) (x,y,score2d)
    cam_params_vec: torch.Tensor,  # (*, V, 48)
    score_threshold: float = 0.3,
    min_error_threshold: float = 16**2,  # squared px error
    img_width: int = 1920,
    img_height: int = 1080,
) -> torch.Tensor:
    """
    Curate pseudo 2D labels using 3D reprojection.
    """
    # --- Split inputs ---
    # 3D: (*, N, J, 4) -> coords + score
    xyz = gt_poses_xyzs[..., :3]  # (*, N, J, 3)
    score3d = gt_poses_xyzs[..., 3]  # (*, N, J)

    # 2D: (*, V, N, J, 3) -> coords + score
    xy_2d = gt_poses_xys[..., :2]  # (*, V, N, J, 2)
    score2d = gt_poses_xys[..., 2]  # (*, V, N, J)

    # --- Validity masks ---
    valid_2d = score2d >= score_threshold  # (*, V, N, J)
    valid_3d = score3d >= score_threshold  # (*, N, J)

    # Broadcast 3D validity to per-view shape
    valid_3d_view = valid_3d.unsqueeze(-3).expand_as(score2d)  # (*, V, N, J)

    # --- Reproject 3D to each view ---
    reproj_xy, reproj_valid = world_3d_to_img_2d(xyz, cam_params_vec)  # (*, V, N, J, 2)

    # --- In-bounds mask for reprojected points ---
    x_rep = reproj_xy[..., 0]
    y_rep = reproj_xy[..., 1]

    in_bounds = (x_rep >= 0) & (x_rep < img_width) & (y_rep >= 0) & (y_rep < img_height)  # (*, V, N, J)

    # 3D is usable only if it exists AND is in-bounds
    valid_3d_usable = valid_3d_view & in_bounds & reproj_valid.squeeze(-1) # (*, V, N, J)

    # --- Reprojection error between existing 2D and 3D reprojection ---
    reproj_err_sq = ((xy_2d - reproj_xy) ** 2).sum(dim=-1)  # (*, V, N, J)

    # --- Decide which source to use per joint ---

    # Case A: 2D is valid AND 3D usable AND error small -> keep 2D
    use_2d = valid_2d & valid_3d_usable & (reproj_err_sq <= min_error_threshold)

    # Case B: otherwise, if 3D is usable -> use 3D
    use_3d = valid_3d_usable & (~use_2d)

    # Case C: 2D valid, 3D valid but out-of-bounds, and they disagree -> Drop
    disagree = reproj_err_sq > min_error_threshold
    drop = valid_2d & valid_3d_view & (~in_bounds) & disagree  # (*, V, N, J)

    # --- Build output ---
    out = gt_poses_xys.clone()
    out_xy = out[..., :2]  # (*, V, N, J, 2)
    out_score = out[..., 2]  # (*, V, N, J)

    # Broadcast 3D scores to per-view shape
    score3d_view = score3d.unsqueeze(-3).expand_as(out_score)  # (*, V, N, J)

    # Replace with 3D reprojection where requested
    out_xy[use_3d] = reproj_xy[use_3d]
    out_score[use_3d] = score3d_view[use_3d]

    # Drop supervision where 2D & 3D disagree but 3D is out-of-bounds
    out_xy[drop] = 0.0
    out_score[drop] = 0.0

    return out

# --------- Loss Functions ---------
class HeatmapLoss(nn.Module):
    """Heatmap loss for keypoint pretraining with sparse pseudo-GT.

    The loss is asymmetric MSE:

        L = mean( relu(target − pred)² + bg_weight · relu(pred − target)² )

    The first term is full-strength supervision for *underprediction at GT
    keypoints* (the model must fire there). The second term is a soft cap on
    *overprediction at non-GT pixels*: small enough that the model isn't
    punished for firing at real-but-unlabeled people, large enough to suppress
    spurious detections at mirrors / reflections / repeated background patterns
    (which look identical to unlabeled people but contribute no real
    information). ``bg_weight=1.0`` recovers standard symmetric MSE; ``bg_weight<1.0``
    applies a soft cap on overprediction so the model is not strongly penalized for
    firing at real-but-unlabeled people while spurious background activations are
    still suppressed.
    """

    def __init__(
        self,
        image_size: tuple[int, int],  # (W, H)
        heatmap_size: tuple[int, int],  # (W, H)
        sigma: float = 2.0,
        use_subpixel_centers: bool = True,
        aggregate_persons: str = "max",  # "max" | "sum" | "or"
        bg_weight: float = 0.1,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.use_subpixel_centers = bool(use_subpixel_centers)
        assert aggregate_persons in ("max", "sum", "or")
        self.aggregate_persons = aggregate_persons
        self.bg_weight = float(bg_weight)

    @staticmethod
    def _apply_affine(points: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
        """
        points: (*, V, N, J, 2)
        affine: (*, V, 2, 3)
        returns: (*, V, N, J, 2)
        """
        ones = torch.ones_like(points[..., :1])
        points_h = torch.cat([points, ones], dim=-1)  # (*, V, N, J, 3)
        affine_exp = affine.unsqueeze(-3).unsqueeze(-3)  # (*, V, 1, 1, 2, 3)
        points_h = points_h.unsqueeze(-1)  # (*, V, N, J, 3, 1)
        return torch.matmul(affine_exp, points_h).squeeze(-1)

    def forward(
        self,
        pred_heatmaps: torch.Tensor,  # (*, V, C=J, H, W)
        gt_poses_xys: torch.Tensor,  # (*, V, N, J, 3)
        gt_poses_xyzs: torch.Tensor,  # (*, N, J, 4) (x,y,z,score3d) — used by process_pseudo_2d
        cam_params_vec: torch.Tensor,  # (*, V, 51)
        affine_transforms: torch.Tensor,  # (*, V, 2, 3)
        **kwargs,
    ) -> torch.Tensor:
        img_width, img_height = original_image_resolution(self.image_size)
        gt_poses_xys = process_pseudo_2d(
            gt_poses_xyzs, gt_poses_xys, cam_params_vec, img_width=img_width, img_height=img_height
        )
        H, W = pred_heatmaps.shape[-2:]

        gt_xy = gt_poses_xys[..., :2]  # (*, V, N, J, 2)
        gt_score = gt_poses_xys[..., 2]  # (*, V, N, J)
        valid_mask = gt_score > 0

        if not valid_mask.any():
            return pred_heatmaps.new_tensor(0.0)

        gt_xy_transformed = self._apply_affine(gt_xy, affine_transforms)
        target_heatmaps = compute_heatmap_vectorized(
            joints=gt_xy_transformed,
            joints_vis=valid_mask,
            sigma=self.sigma,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            per_person=False,
            joint_weights=None,
            use_subpixel_centers=self.use_subpixel_centers,
            aggregate_persons=self.aggregate_persons,
        )  # (*, V, J, H, W)

        # Supervise only (view, joint) that has at least one visible person
        joint_has_supervision = valid_mask.any(dim=-2)  # (*, V, J)
        num_supervised = joint_has_supervision.sum().to(pred_heatmaps.dtype)
        if num_supervised <= 0:
            return pred_heatmaps.new_tensor(0.0)

        # Asymmetric MSE: full weight for underprediction, bg_weight for overprediction.
        diff = target_heatmaps - pred_heatmaps
        underpred = torch.clamp(diff, min=0.0)
        overpred = torch.clamp(-diff, min=0.0)
        sq = underpred * underpred + self.bg_weight * (overpred * overpred)

        sq = sq * joint_has_supervision[..., None, None]  # (*, V, J, H, W)
        return sq.sum() / (num_supervised * H * W)
    
    
class TriangulationResidualLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, triangulation_residuals: torch.Tensor | list[torch.Tensor], mask: torch.Tensor, **kwargs):
        if isinstance(triangulation_residuals, list):
            triangulation_residuals = triangulation_residuals[0]
        if isinstance(mask, list):
            mask = mask[0]
        _sum = (triangulation_residuals * mask).sum()
        _total = mask.sum()

        if _total <= 0:
            return triangulation_residuals.new_tensor(0.0)

        return _sum / _total


class PseudoRegressionCrossAffineL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def _single_branch_loss(
        self,
        pred_xyz: torch.Tensor,
        person_ids: torch.Tensor,
        dim_A: int,
    ) -> torch.Tensor:
        # Move A dimension to the front
        pred_xyz = pred_xyz.movedim(dim_A, 0)  # (A, ..., N, J, 3)
        person_ids = person_ids.movedim(dim_A, 0)  # (A, ..., N)

        A = pred_xyz.shape[0]
        if A <= 1:
            return pred_xyz.new_tensor(0.0)

        xyz_shape = pred_xyz.shape
        N, J, C = xyz_shape[-3:]

        # Flatten all leading dims except A and (N, J, 3)
        if len(xyz_shape) > 4:
            R = math.prod(xyz_shape[1:-3])
        else:
            R = 1

        pred_xyz_flat = pred_xyz.reshape(A, R, N, J, C)
        person_ids_flat = person_ids.reshape(A, R, N)

        total_l1 = pred_xyz.new_tensor(0.0)
        total_coords = pred_xyz.new_tensor(0.0)

        for a1, a2 in combinations(range(A), 2):
            ids1 = person_ids_flat[a1]
            ids2 = person_ids_flat[a2]

            valid_persons = (ids1 == ids2) & (ids1 != -1)  # (R, N)
            if not valid_persons.any():
                continue

            xyz1 = pred_xyz_flat[a1]
            xyz2 = pred_xyz_flat[a2]

            valid_mask = valid_persons.unsqueeze(-1).unsqueeze(-1)
            diff = (xyz1 - xyz2).abs() * valid_mask

            sum_l1 = diff.sum()
            num_pairs = valid_persons.sum()
            num_coords = num_pairs * J * C

            total_l1 = total_l1 + sum_l1
            total_coords = total_coords + num_coords.to(dtype=total_l1.dtype)

        if total_coords == 0:
            return pred_xyz.new_tensor(0.0)

        return total_l1 / total_coords

    def forward(
        self,
        pred_xyz: list[torch.Tensor] | torch.Tensor,
        person_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        dim_A = kwargs["dim_A"]
        if isinstance(pred_xyz, torch.Tensor):
            pred_xyz = [pred_xyz]
        if len(pred_xyz) == 0:
            raise ValueError("pred_xyz must contain at least one tensor")

        loss = pred_xyz[0].new_tensor(0.0)
        for branch_xyz in pred_xyz:
            loss = loss + self._single_branch_loss(branch_xyz, person_ids, dim_A)

        return loss


class Pseudo3DRegressionL1Loss(nn.Module):
    def __init__(self, gt_threshold: float, conf_gamma: float = 1.0):
        super().__init__()
        self.gt_threshold = gt_threshold
        self.conf_gamma = conf_gamma

    @staticmethod
    def _branch_l1_3d(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(pred_xyz.dtype)
        weights_unsq = weights.unsqueeze(-1)
        diff = (pred_xyz - gt_xyz).abs() * weights_unsq
        total = diff.sum()
        denom = weights.sum() * 3.0
        if denom <= 0:
            return pred_xyz.new_tensor(0.0)
        return total / denom

    def forward(
        self,
        pred_poses_xyz: list[torch.Tensor] | torch.Tensor,
        gt_poses_xyzs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(pred_poses_xyz, torch.Tensor):
            pred_poses_xyz = [pred_poses_xyz]
        if len(pred_poses_xyz) == 0:
            raise ValueError("pred_poses_xyz must contain at least one tensor")

        gt_xyz = gt_poses_xyzs[..., :3]
        gt_score = gt_poses_xyzs[..., 3]

        joints_vis = gt_score >= self.gt_threshold
        if not joints_vis.any():
            return pred_poses_xyz[0].new_tensor(0.0)

        conf = gt_score.clamp(min=0.0, max=1.0)
        if self.conf_gamma != 1.0:
            conf = conf**self.conf_gamma
        weights = conf * joints_vis.to(conf.dtype)

        loss = pred_poses_xyz[0].new_tensor(0.0)
        for pred_xyz in pred_poses_xyz:
            if pred_xyz.shape[-3:-1] != gt_xyz.shape[-3:-1]:
                raise ValueError(f"Shape mismatch: {pred_xyz.shape} vs {gt_xyz.shape}")
            loss = loss + self._branch_l1_3d(pred_xyz, gt_xyz, weights)

        return loss


class PseudoRegressionL1Loss(nn.Module):
    def __init__(self, gt_threshold: float, conf_gamma: float = 1.0):
        super().__init__()
        self.gt_threshold = gt_threshold
        self.conf_gamma = conf_gamma

    @staticmethod
    def _branch_l1(pred_xy: torch.Tensor, gt_xy: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(pred_xy.dtype)
        weights_unsq = weights.unsqueeze(-1)
        diff = (pred_xy - gt_xy).abs() * weights_unsq
        total = diff.sum()
        denom = weights.sum() * 2.0
        if denom <= 0:
            return pred_xy.new_tensor(0.0)
        return total / denom

    def forward(
        self,
        pred_poses_xy: list[torch.Tensor] | torch.Tensor,
        gt_poses_xys: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(pred_poses_xy, torch.Tensor):
            pred_poses_xy = [pred_poses_xy]
        if len(pred_poses_xy) == 0:
            raise ValueError("pred_poses_xy must contain at least one tensor")

        gt_xy = gt_poses_xys[..., :2]
        gt_score = gt_poses_xys[..., 2]

        valid_mask = gt_score >= self.gt_threshold
        if not valid_mask.any():
            return pred_poses_xy[0].new_tensor(0.0)

        conf = gt_score.clamp(min=0.0, max=1.0)
        if self.conf_gamma != 1.0:
            conf = conf**self.conf_gamma
        weights = conf * valid_mask.to(conf.dtype)

        loss = pred_poses_xy[0].new_tensor(0.0)
        for pred_xy in pred_poses_xy:
            loss = loss + self._branch_l1(pred_xy, gt_xy, weights)
        return loss


class PseudoRegressionL2Loss(nn.Module):
    """
    L2 HeatMap Loss with Dynamic Normalization and Differentiable Gaussian Generation.
    """

    def __init__(
        self,
        gt_threshold: float,
        sigma: float = 2.0,
        conf_gamma: float = 1.0,
        image_size: tuple[int, int] = (960, 512),
        heatmap_size: tuple[int, int] = (240, 128),
    ):
        super().__init__()
        self.gt_threshold = gt_threshold
        self.sigma = float(sigma)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.conf_gamma = conf_gamma

    @staticmethod
    def _apply_affine(points: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(points[..., :1])
        points_h = torch.cat([points, ones], dim=-1)
        affine_exp = affine.unsqueeze(-3).unsqueeze(-3)
        points_h = points_h.unsqueeze(-1)
        transformed = torch.matmul(affine_exp, points_h).squeeze(-1)
        return transformed

    @staticmethod
    def _weighted_one_sided_l2(pred_hm, target_hm, weights):
        H, W = pred_hm.shape[-2:]
        normalization_factor = weights.sum() * (H * W)
        if normalization_factor <= 1e-6:
            return pred_hm.new_tensor(0.0)
        # target_hm is GT-derived (no grad needed back through it). Use the custom
        # autograd Function below to retain only `diff = relu(target - pred)` and the
        # weights tensor for backward — naive autograd would also save the per-element
        # sq_error and weighted_sq_error intermediates (each ~1 GiB at bs=4 with the
        # per-person heatmap shape), tripling the chain's working set.
        return _OneSidedL2.apply(pred_hm, target_hm, weights, normalization_factor)

    def prepare_target(
        self,
        gt_poses_xys: torch.Tensor,
        affine_transforms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Build the GT-only target heatmap and masks.

        target_hm depends on GT alone; for deep-supervised stages with identical GT
        across stages, callers can compute this once and pass it back via ``target_cache``
        in :meth:`forward` to avoid recomputing the same ~hundred-MB tensor per stage.
        Returns ``None`` if no joint is valid in this batch (caller short-circuits to 0).
        """
        gt_xy = gt_poses_xys[..., :2]
        gt_score = gt_poses_xys[..., 2]
        valid_mask = gt_score >= self.gt_threshold
        if not valid_mask.any():
            return None

        conf = gt_score.clamp(min=0.0, max=1.0)
        if self.conf_gamma != 1.0:
            conf = conf**self.conf_gamma
        loss_weights = conf * valid_mask.to(conf.dtype)

        gt_xy_in = self._apply_affine(gt_xy, affine_transforms)
        target_hm = compute_heatmap_vectorized(
            joints=gt_xy_in,
            joints_vis=valid_mask,
            sigma=self.sigma,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            per_person=True,
            joint_weights=loss_weights,
        )
        return target_hm, valid_mask, loss_weights

    def forward(
        self,
        pred_poses_xy: list[torch.Tensor] | torch.Tensor,
        gt_poses_xys: torch.Tensor,
        affine_transforms: torch.Tensor,
        target_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(pred_poses_xy, torch.Tensor):
            pred_poses_xy = [pred_poses_xy]
        if len(pred_poses_xy) == 0:
            raise ValueError("pred_poses_xy must contain at least one tensor")

        if target_cache is None:
            target_cache = self.prepare_target(gt_poses_xys, affine_transforms)
            if target_cache is None:
                return pred_poses_xy[0].new_tensor(0.0)
        target_hm, valid_mask, loss_weights = target_cache

        loss = pred_poses_xy[0].new_tensor(0.0)
        for pred_xy in pred_poses_xy:
            pred_xy_in = self._apply_affine(pred_xy, affine_transforms)
            pred_hm = compute_heatmap_vectorized(
                joints=pred_xy_in,
                joints_vis=valid_mask,
                sigma=self.sigma,
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                per_person=True,
                joint_weights=None,
            )
            loss = loss + self._weighted_one_sided_l2(pred_hm, target_hm, loss_weights)

        return loss


# --- Utility Functions ---
def compute_heatmap_vectorized(
    joints: torch.Tensor,  # (*, V, N, J, 2)
    joints_vis: torch.Tensor,  # (*, V, N, J)
    sigma: float,
    image_size: tuple[int, int],  # (W, H)
    heatmap_size: tuple[int, int],  # (W, H)
    per_person: bool = False,
    joint_weights: torch.Tensor | None = None,
    use_subpixel_centers: bool = True,
    aggregate_persons: str = "max",  # "max" | "sum" | "or"
) -> torch.Tensor:
    device = joints.device
    dtype = joints.dtype
    assert aggregate_persons in ("max", "sum", "or")

    vis_mask = joints_vis > 0.5 if joints_vis.dtype != torch.bool else joints_vis

    *batch_dims, nposes, njoints, _ = joints.shape  # batch_dims includes V here
    heatmap_w, heatmap_h = heatmap_size
    img_w, img_h = image_size

    feat_stride_x = img_w / heatmap_w
    feat_stride_y = img_h / heatmap_h

    joints_hm_x = joints[..., 0] / feat_stride_x
    joints_hm_y = joints[..., 1] / feat_stride_y

    if use_subpixel_centers:
        mu_x = joints_hm_x.unsqueeze(-1).unsqueeze(-1)
        mu_y = joints_hm_y.unsqueeze(-1).unsqueeze(-1)
        margin = float(sigma * 3.0)
        in_bounds = (
            (joints_hm_x >= -margin)
            & (joints_hm_x < heatmap_w + margin)
            & (joints_hm_y >= -margin)
            & (joints_hm_y < heatmap_h + margin)
        )
    else:
        mu_x_int = joints_hm_x.round().long()
        mu_y_int = joints_hm_y.round().long()
        mu_x = mu_x_int.to(dtype).unsqueeze(-1).unsqueeze(-1)
        mu_y = mu_y_int.to(dtype).unsqueeze(-1).unsqueeze(-1)
        margin = int(sigma * 3)
        in_bounds = (
            (mu_x_int >= -margin)
            & (mu_x_int < heatmap_w + margin)
            & (mu_y_int >= -margin)
            & (mu_y_int < heatmap_h + margin)
        )

    valid = vis_mask & in_bounds  # (*, V, N, J)

    y_coords = torch.arange(heatmap_h, device=device, dtype=dtype)
    x_coords = torch.arange(heatmap_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    grid_shape = [1] * len(batch_dims) + [1, 1, heatmap_h, heatmap_w]
    xx = xx.view(*grid_shape)
    yy = yy.view(*grid_shape)

    dist_sq = (xx - mu_x) ** 2 + (yy - mu_y) ** 2
    gaussian = torch.exp(-dist_sq / (2.0 * sigma * sigma))

    gaussian = gaussian * valid.unsqueeze(-1).unsqueeze(-1).to(dtype)

    if joint_weights is not None:
        jw = joint_weights.clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
        gaussian = gaussian * jw

    if per_person:
        return gaussian.clamp(0.0, 1.0)

    # aggregate across persons: dimension -4 in (..., N, J, H, W)
    if aggregate_persons == "max":
        out = gaussian.max(dim=-4).values
    elif aggregate_persons == "sum":
        out = gaussian.sum(dim=-4).clamp(0.0, 1.0)
    else:  # "or"
        out = (1.0 - (1.0 - gaussian).prod(dim=-4)).clamp(0.0, 1.0)

    return out.clamp(0.0, 1.0)