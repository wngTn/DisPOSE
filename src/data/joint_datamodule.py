from omegaconf import DictConfig

from src.utils.common import is_tensor, np2ts

from .base import BASEDataModule


class JointDataModule(BASEDataModule):
    """DataModule for multi-view joint pose estimation with ImageNet normalization."""

    def __init__(
        self,
        train_cfg: DictConfig,
        val_cfg: DictConfig,
        test_cfg: DictConfig | None = None,
    ) -> None:
        super().__init__(train_cfg, val_cfg, test_cfg)
        self.njoints = 15
        self.name = "Joint"

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.Dataset = {
            "train": train_cfg.dataset,
            "val": val_cfg.dataset,
        }

    def denormalize(self, x):
        if self.mean is None and self.std is None:
            return x
        if is_tensor(x):
            std = np2ts(self.std, dtype=x.dtype, device=x.device)
            mean = np2ts(self.mean, dtype=x.dtype, device=x.device)
        else:
            std = self.std
            mean = self.mean
        return (x * std[: x.shape[-1]]) + mean[: x.shape[-1]]
