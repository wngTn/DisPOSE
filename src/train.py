import logging
import os
from typing import Any
import warnings
from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig

warnings.filterwarnings("ignore", category=UserWarning, module="mmcv")
warnings.filterwarnings("ignore", message=".*tensorboardX.*")
warnings.filterwarnings("ignore", message=".*litlogger.*")
warnings.filterwarnings("ignore", message=".*litmodels.*")
torch.set_float32_matmul_precision("high")

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.utils.resume import resume_experiment

log = logging.getLogger(__name__)


def _load_init_checkpoint(cfg: DictConfig, model: L.LightningModule) -> None:
    init_ckpt_path = cfg.get("init_ckpt_path")
    if not init_ckpt_path:
        return

    ckpt_path = Path(init_ckpt_path)
    log.info("Initializing model weights from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.info("Missing keys while loading init checkpoint: %d", len(missing))
    if unexpected:
        log.info("Unexpected keys while loading init checkpoint: %d", len(unexpected))


def _allow_resume_checkpoint_globals() -> None:
    """Allowlist OmegaConf containers for torch>=2.6 checkpoint loading.

    Lightning checkpoint resume relies on ``torch.load`` under the hood. With
    torch 2.6 the default ``weights_only=True`` path rejects OmegaConf objects
    stored in the checkpoint metadata unless they are explicitly allowlisted.
    """
    torch.serialization.add_safe_globals([Any, ContainerMetadata, OmegaDictConfig, OmegaListConfig])


def _silence_non_rank_zero_file_logging() -> None:
    """In DDP, every spawned process re-enters @hydra.main and creates its own
    timestamped .log file. Keep only rank 0's; remove the file handler on the
    others so we end up with a single log file per run.
    """
    if os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0":
        return
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            stream_path = Path(handler.baseFilename)
            root.removeHandler(handler)
            handler.close()
            if stream_path.exists() and stream_path.stat().st_size == 0:
                stream_path.unlink()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    _silence_non_rank_zero_file_logging()
    cfg = resume_experiment(cfg)
    _allow_resume_checkpoint_globals()

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating model")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    _load_init_checkpoint(cfg, model)

    log.info("Instantiating datamodule")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    callbacks = []
    if cfg.get("callbacks"):
        for cb_name, cb_cfg in cfg.callbacks.items():
            if cb_cfg is not None and "_target_" in cb_cfg:
                callbacks.append(hydra.utils.instantiate(cb_cfg))

    logger = None
    if cfg.get("logger") and cfg.task_name != "debug":
        logger = hydra.utils.instantiate(cfg.logger)

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.get("deterministic", False),
    )

    # Resume from checkpoint if specified
    ckpt_path = cfg.get("ckpt_path")

    log.info("Starting training")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        weights_only=False if ckpt_path else None,
    )
    log.info("Training finished — step %d", trainer.global_step)


if __name__ == "__main__":
    main()
