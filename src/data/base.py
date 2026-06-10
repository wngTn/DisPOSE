import logging
from typing import Any

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class BASEDataModule(L.LightningDataModule):
    """Base data module for multi-view pose estimation datasets.

    Wraps train/val/test dataset creation and dataloader configuration.
    Datasets are lazily instantiated on first access.
    """

    def __init__(
        self,
        train_cfg: DictConfig,
        val_cfg: DictConfig,
        test_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.test_cfg = test_cfg

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None

        # Subclass must set these
        self.Dataset: dict[str, Any] = {}
        self.mean: list[float] = []
        self.std: list[float] = []

    # --- Lazy dataset properties ---

    @property
    def train_dataset(self) -> Dataset:
        if self._train_dataset is None:
            self._train_dataset = self.Dataset["train"](split="train", mean=self.mean, std=self.std)
        return self._train_dataset  # type: ignore

    @property
    def val_dataset(self) -> Dataset:
        if self._val_dataset is None:
            self._val_dataset = self.Dataset["val"](split="test", mean=self.mean, std=self.std)
        return self._val_dataset  # type: ignore

    @property
    def test_dataset(self) -> Dataset:
        return self.val_dataset

    # --- Lightning hooks ---

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
            log.info(f"Train dataset: {len(self.train_dataset)} samples")  # type: ignore
        if stage in (None, "test", "predict", "validate"):
            _ = self.val_dataset
            log.info(f"Val/test dataset: {len(self.val_dataset)} samples")  # type: ignore

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_cfg["batch_sampler"](self.train_dataset),
            collate_fn=self.train_dataset.collate_fn,  # type: ignore
            **self.train_cfg.loader_params,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.val_cfg["batch_sampler"](self.val_dataset),
            collate_fn=self.val_dataset.collate_fn,  # type: ignore
            **self.val_cfg.loader_params,
        )

    def test_dataloader(self) -> DataLoader:
        cfg = self.test_cfg if self.test_cfg is not None else self.val_cfg
        dataset = self.test_dataset
        return DataLoader(
            dataset,
            batch_sampler=cfg["batch_sampler"](dataset),
            collate_fn=dataset.collate_fn,  # type: ignore
            **cfg.loader_params,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
