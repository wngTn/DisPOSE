"""Heatmap-overlay visualization for backbone validation."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.callbacks import Callback


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _denormalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalized (C, H, W) → uint8 RGB (H, W, 3)."""
    mean = _IMAGENET_MEAN.to(x.device).view(3, 1, 1)
    std = _IMAGENET_STD.to(x.device).view(3, 1, 1)
    rgb = (x * std + mean).clamp(0.0, 1.0)
    return (rgb.permute(1, 2, 0) * 255.0).round().to(torch.uint8)


def _inferno(values: torch.Tensor) -> torch.Tensor:
    """(H, W) values in [0, 1] → uint8 inferno RGB (H, W, 3)."""
    import matplotlib

    cmap = matplotlib.colormaps["inferno"]
    lut = torch.tensor(cmap.colors, dtype=torch.float32, device=values.device)  # (256, 3)
    idx = (values.clamp(0.0, 1.0) * 255.0).round().long()
    return (lut[idx] * 255.0).round().to(torch.uint8)


class HeatmapVisualizationCallback(Callback):
    """Overlay predicted heatmaps onto multi-view RGB during validation.

    Reservoir-samples ``num_frames`` multi-view frames uniformly at random from
    the rank-0 val stream each epoch. For each selected frame, max-pools the
    heatmap across the joint dimension to a single activation map per view,
    upsamples to image resolution, and blends with the inferno colormap
    (alpha = ``alpha``). Saves to ``${trainer.default_root_dir}/viz/step-NNNNNN/``
    and logs to W&B when a :class:`WandbLogger` is active.
    """

    def __init__(self, num_frames: int = 20, alpha: float = 0.55, sample_seed: int | None = None):
        super().__init__()
        self.num_frames = int(num_frames)
        self.alpha = float(alpha)
        self.sample_seed = sample_seed
        self._frames: list[torch.Tensor] = []
        self._seen: int = 0
        self._rng: random.Random | None = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._frames = []
        self._seen = 0
        seed = (self.sample_seed if self.sample_seed is not None else 0) ^ int(trainer.global_step)
        self._rng = random.Random(seed)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.is_global_zero:
            return
        if not isinstance(outputs, dict) or "heatmaps" not in outputs:
            return

        # heatmaps: (B, A, T, V, J, H_hm, W_hm); sources: (B, A, T, V, C, H_img, W_img)
        hm = rearrange(outputs["heatmaps"], "b a t v j h w -> (b a t) v j h w")
        src = rearrange(batch["source"], "b a t v c h w -> (b a t) v c h w")

        for f in range(hm.shape[0]):
            self._reservoir_consider(src[f], hm[f])

    def _reservoir_consider(self, src_v: torch.Tensor, hm_v: torch.Tensor) -> None:
        """Reservoir-sample N frames uniformly from a stream; render only on selection."""
        assert self._rng is not None
        i = self._seen
        self._seen += 1
        if i < self.num_frames:
            self._frames.append(self._render_frame(src_v, hm_v).cpu())
            return
        j = self._rng.randint(0, i)
        if j < self.num_frames:
            self._frames[j] = self._render_frame(src_v, hm_v).cpu()

    def _render_frame(self, src_v: torch.Tensor, hm_v: torch.Tensor) -> torch.Tensor:
        """src_v: (V, C, H_img, W_img); hm_v: (V, J, H_hm, W_hm) → (H_img, V·W_img, 3) uint8."""
        V, _, H_img, W_img = src_v.shape

        act = hm_v.amax(dim=1, keepdim=True)  # max over joints → (V, 1, H_hm, W_hm)
        act = F.interpolate(act, size=(H_img, W_img), mode="bilinear", align_corners=False).squeeze(1)
        # Per-view min-max normalize so the colormap range is meaningful even early in training
        act_min = act.amin(dim=(1, 2), keepdim=True)
        act_max = act.amax(dim=(1, 2), keepdim=True)
        act = (act - act_min) / (act_max - act_min).clamp_min(1e-6)

        views = []
        for v in range(V):
            rgb = _denormalize(src_v[v]).float()
            heat = _inferno(act[v].float()).float()
            views.append((rgb * (1.0 - self.alpha) + heat * self.alpha).round().clamp(0, 255).to(torch.uint8))
        return torch.cat(views, dim=1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or not self._frames:
            return

        step = int(trainer.global_step)
        save_dir = Path(trainer.default_root_dir) / "viz" / f"step-{step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        for i, frame in enumerate(self._frames):
            Image.fromarray(frame.numpy()).save(save_dir / f"frame_{i:02d}.jpg", quality=85)

        # Log to W&B if active.
        try:
            from lightning.pytorch.loggers import WandbLogger
            import wandb
        except ImportError:
            return
        for logger in trainer.loggers or []:
            if isinstance(logger, WandbLogger):
                images = [
                    wandb.Image(frame.numpy(), caption=f"frame_{i:02d}")
                    for i, frame in enumerate(self._frames)
                ]
                logger.experiment.log({"val/heatmap_viz": images, "trainer/global_step": step})
                break
