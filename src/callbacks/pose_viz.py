"""Multi-view 3D-pose-overlay visualization for pose-task validation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from lightning.pytorch.callbacks import Callback

from src.utils.camera import world_3d_to_img_2d
from src.utils.linear_algebra import affine_transform


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# 15-joint panoptic skeleton edges (parent → child); see KEYPOINT_INFO in paramUtil.
_PANOPTIC_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),                    # neck → nose
    (0, 2),                    # neck → mid_hip
    (0, 3), (3, 4), (4, 5),    # neck → left arm
    (0, 9), (9, 10), (10, 11), # neck → right arm
    (2, 6), (6, 7), (7, 8),    # mid_hip → left leg
    (2, 12), (12, 13), (13, 14),  # mid_hip → right leg
)


# Tab-10 person palette (RGB 0-255).
_PERSON_COLORS = np.array(
    [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207),
    ],
    dtype=np.uint8,
)


def _denormalize(x: torch.Tensor) -> np.ndarray:
    """ImageNet-normalized (C, H, W) → uint8 RGB (H, W, 3) numpy."""
    mean = _IMAGENET_MEAN.to(x.device).view(3, 1, 1)
    std = _IMAGENET_STD.to(x.device).view(3, 1, 1)
    rgb = (x * std + mean).clamp(0.0, 1.0)
    return (rgb.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()


def _draw_circle(img: np.ndarray, x: int, y: int, color: np.ndarray, radius: int = 4) -> None:
    """Filled-circle overlay; clipped to image bounds."""
    H, W, _ = img.shape
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = xx * xx + yy * yy <= radius * radius
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    sub_mask = mask[y0 - (y - radius) : (y1 - y) + radius, x0 - (x - radius) : (x1 - x) + radius]
    img[y0:y1, x0:x1][sub_mask] = color


def _draw_line(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: np.ndarray, thickness: int = 2) -> None:
    """Bresenham line, thickened by stamping circles along the path."""
    x0, y0 = p0
    x1, y1 = p1
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    if n <= 0:
        return
    xs = np.linspace(x0, x1, n).round().astype(np.int64)
    ys = np.linspace(y0, y1, n).round().astype(np.int64)
    for x, y in zip(xs, ys):
        _draw_circle(img, int(x), int(y), color, radius=thickness)


class PoseVisualizationCallback(Callback):
    """Overlay predicted 3D poses (final regressor stage) onto multi-view RGB.

    Reservoir-samples ``num_frames`` validation frames uniformly at random from
    the rank-0 val stream each epoch (rather than just taking the first N), then
    reprojects the final-stage refined 3D poses to each view via cam params +
    source-image affine and draws a colored skeleton over the input image.
    Saves to ``${trainer.default_root_dir}/viz/step-NNNNNN/`` and logs to W&B
    when a :class:`WandbLogger` is active.
    """

    def __init__(self, num_frames: int = 20, score_threshold: float = 0.0, sample_seed: int | None = None):
        super().__init__()
        self.num_frames = int(num_frames)
        self.score_threshold = float(score_threshold)
        self.sample_seed = sample_seed
        self._frames: list[np.ndarray] = []
        self._seen: int = 0
        self._rng: random.Random | None = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._frames = []
        self._seen = 0
        # Vary the sample across val epochs but stay deterministic given a fixed seed + step.
        seed = (self.sample_seed if self.sample_seed is not None else 0) ^ int(trainer.global_step)
        self._rng = random.Random(seed)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.is_global_zero:
            return
        if not isinstance(outputs, dict) or "refined_poses_xyz" not in outputs:
            return

        # Take the final stage of refined poses.
        # refined_poses_xyz: (B, A, T, L, N, J, 3) → (B, A, T, N, J, 3)
        refined = outputs["refined_poses_xyz"][..., -1, :, :, :]
        sources = batch["source"]                      # (B, A, T, V, C, H, W)
        cam_params = batch["cam_params_vec"]            # (B, A, T, V, CAM_DIM)
        affines = batch["affine_transforms"]            # (B, A, T, V, 2, 3)

        # Decode confidence per person from the assignment net (if present).
        scores: torch.Tensor | None = None
        assn = outputs.get("assignment_xyzs")
        if assn is not None:
            scores = assn[..., -1]                      # (B, A, T, N)

        # Collapse (B, A, T) into a frame index.
        refined_f = rearrange(refined, "b a t n j d -> (b a t) n j d")
        src_f = rearrange(sources, "b a t v c h w -> (b a t) v c h w")
        cam_f = rearrange(cam_params, "b a t v d -> (b a t) v d")
        aff_f = rearrange(affines, "b a t v r c -> (b a t) v r c")
        scores_f = rearrange(scores, "b a t n -> (b a t) n") if scores is not None else None

        for f in range(refined_f.shape[0]):
            score_f = scores_f[f] if scores_f is not None else None
            self._reservoir_consider(src_f[f], refined_f[f], score_f, cam_f[f], aff_f[f])

    def _reservoir_consider(
        self,
        src_v: torch.Tensor,
        pred_xyz: torch.Tensor,
        scores: torch.Tensor | None,
        cam_params: torch.Tensor,
        affines: torch.Tensor,
    ) -> None:
        """Standard reservoir sampling: keep N uniformly-random frames from a stream.

        Renders only when the frame is actually selected, so rejected candidates
        cost effectively nothing.
        """
        assert self._rng is not None
        i = self._seen
        self._seen += 1
        if i < self.num_frames:
            self._frames.append(self._render_frame(src_v, pred_xyz, scores, cam_params, affines))
            return
        j = self._rng.randint(0, i)  # inclusive upper bound = i, the standard formulation
        if j < self.num_frames:
            self._frames[j] = self._render_frame(src_v, pred_xyz, scores, cam_params, affines)

    def _render_frame(
        self,
        src_v: torch.Tensor,           # (V, C, H_img, W_img)
        pred_xyz: torch.Tensor,         # (N, J, 3) world coords
        scores: torch.Tensor | None,    # (N,) per-person score
        cam_params: torch.Tensor,       # (V, CAM_DIM)
        affines: torch.Tensor,          # (V, 2, 3)
    ) -> np.ndarray:
        V, _, H_img, W_img = src_v.shape
        N, J, _ = pred_xyz.shape

        # Reproject world → original-image space.
        # world_3d_to_img_2d expects X: (..., N, J, 3), cam: (..., V, CAM_DIM)
        # We want a (V, N, J, 2) result.
        uv_orig, valid = world_3d_to_img_2d(
            pred_xyz.unsqueeze(0),  # (1, N, J, 3)
            cam_params.unsqueeze(0),  # (1, V, CAM_DIM)
        )
        uv_orig = uv_orig[0]      # (V, N, J, 2)
        valid = valid[0, ..., 0]  # (V, N, J) bool

        # Original-image space → source/input-image space.
        # affine_transform: (V, *, 2) + (V, 2, 3) → (V, *, 2)
        uv_src = affine_transform(rearrange(uv_orig, "v n j d -> v (n j) d"), affines)
        uv_src = rearrange(uv_src, "v (n j) d -> v n j d", n=N).cpu().numpy()
        valid_np = valid.cpu().numpy()

        if scores is not None:
            person_keep = (scores > self.score_threshold).cpu().numpy()
        else:
            person_keep = np.ones(N, dtype=bool)

        views = []
        for v in range(V):
            img = _denormalize(src_v[v]).copy()
            for n in range(N):
                if not person_keep[n]:
                    continue
                color = _PERSON_COLORS[n % len(_PERSON_COLORS)]
                joints = uv_src[v, n]                  # (J, 2)
                vis = valid_np[v, n] & np.all(np.isfinite(joints), axis=-1)
                # Skeleton edges first (drawn under joints).
                for a, b in _PANOPTIC_EDGES:
                    if a >= J or b >= J or not (vis[a] and vis[b]):
                        continue
                    _draw_line(img, (int(joints[a, 0]), int(joints[a, 1])),
                                  (int(joints[b, 0]), int(joints[b, 1])), color, thickness=2)
                # Joint dots.
                for j in range(J):
                    if vis[j]:
                        _draw_circle(img, int(joints[j, 0]), int(joints[j, 1]), color, radius=4)
            views.append(img)
        return np.concatenate(views, axis=1)  # (H_img, V·W_img, 3) uint8

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or not self._frames:
            return

        step = int(trainer.global_step)
        save_dir = Path(trainer.default_root_dir) / "viz" / f"step-{step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        for i, frame in enumerate(self._frames):
            Image.fromarray(frame).save(save_dir / f"frame_{i:02d}.jpg", quality=85)

        try:
            from lightning.pytorch.loggers import WandbLogger
            import wandb
        except ImportError:
            return
        for logger in trainer.loggers or []:
            if isinstance(logger, WandbLogger):
                images = [wandb.Image(frame, caption=f"frame_{i:02d}") for i, frame in enumerate(self._frames)]
                logger.experiment.log({"val/pose_viz": images, "trainer/global_step": step})
                break
