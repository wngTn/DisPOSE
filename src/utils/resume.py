import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _checkpoint_candidates(run_path: Path, hint: int | str) -> list[Path]:
    """Ordered checkpoint candidates for a restore hint.

    ``hint`` may be:
      - ``"last"``  -> the rolling ``last.ckpt`` (the latest training state),
      - ``"best"``  -> the highest-step ``best-*.ckpt``, falling back to ``last.ckpt``,
      - an integer step -> the checkpoint saved at that step (several name formats).
    """
    checkpoints_dir = run_path / "checkpoints"
    hint_str = str(hint).strip().lower()

    if hint_str == "last":
        return [checkpoints_dir / "last.ckpt"]
    if hint_str == "best":
        best = sorted(checkpoints_dir.glob("best-*.ckpt"), reverse=True)
        return best + [checkpoints_dir / "last.ckpt"]

    step = int(hint)
    return [
        checkpoints_dir / f"best-step={step:06d}.ckpt",
        checkpoints_dir / f"step={step:06d}.ckpt",
        checkpoints_dir / f"{step:06d}.ckpt",
        checkpoints_dir / f"{step:06d}_iteration.pt",
    ]


def resume_experiment(cfg: DictConfig) -> DictConfig:
    """Resume from a previous run directory.

    If +restore_from (dir) is provided:
      1. Loads the original config from that directory.
      2. Sets ckpt_path based on +restore_hint.
      3. Merges ONLY valid CLI overrides onto the old config.
    """
    resume_dir = cfg.get("restore_from")
    if not resume_dir:
        return cfg

    current_output_dir = OmegaConf.select(cfg, "paths.output_dir")
    run_path = Path(resume_dir)
    restore_hint = cfg.get("restore_hint") or "last"

    log.info("Resuming run from %s", run_path)

    # Load the old configuration
    old_config_path = run_path / ".hydra" / "config.yaml"
    if not old_config_path.exists():
        old_config_path = run_path / "config.yaml"
    if not old_config_path.exists():
        raise FileNotFoundError(f"Cannot find config.yaml in {run_path}")

    resume_cfg = OmegaConf.load(old_config_path)

    # Resolve checkpoint path
    if restore_hint:
        # Try Lightning format first, then legacy
        ckpt_path = None
        for pattern in _checkpoint_candidates(run_path, restore_hint):
            if pattern.exists():
                ckpt_path = pattern
                break

        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for hint={restore_hint} in {run_path / 'checkpoints'}"
            )
        OmegaConf.update(resume_cfg, "ckpt_path", str(ckpt_path))
        log.info("Resuming from checkpoint %s", ckpt_path.name)

    # Extract CLI overrides
    try:
        hydra_overrides = HydraConfig.get().overrides.task
    except ValueError:
        # No active Hydra run (HydraConfig is not initialized).
        hydra_overrides = []

    clean_dotlist = []
    for override in hydra_overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            clean_key = key.lstrip("+~")
            clean_dotlist.append(f"{clean_key}={value}")

    cli_cfg = OmegaConf.from_dotlist(clean_dotlist)

    # Merge
    merged_cfg = OmegaConf.merge(resume_cfg, cli_cfg)

    original_task_name = OmegaConf.select(resume_cfg, "task_name")
    requested_task_name = OmegaConf.select(cli_cfg, "task_name")
    resume_in_place = requested_task_name in (None, original_task_name)

    if resume_in_place:
        merged_cfg.paths.output_dir = resume_dir
    elif current_output_dir is not None:
        merged_cfg.paths.output_dir = current_output_dir

    # Reconfigure Hydra log handler to the resumed path
    if resume_in_place:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                current_log_path = Path(handler.baseFilename)
                new_log_path = run_path / current_log_path.name
                if current_log_path != new_log_path:
                    handler.close()
                    if current_log_path.exists():
                        shutil.move(str(current_log_path), str(new_log_path))
                    handler.baseFilename = str(new_log_path)
                    handler.stream = open(new_log_path, "a")

    return merged_cfg  # type: ignore


def _override_camera_setup(cfg, camera_setup: str) -> None:
    """Recursively replace every ``camera_setup`` value in an OmegaConf dict config."""
    if not OmegaConf.is_dict(cfg):
        return
    for key in cfg:
        if key == "camera_setup":
            cfg[key] = camera_setup
        elif OmegaConf.is_dict(cfg[key]):
            _override_camera_setup(cfg[key], camera_setup)


def load_pretrained_component(
    experiment_spec: tuple[str, int],
    component_type: str,
    logs_root: str | Path = "./logs",
    device: torch.device | str = "cpu",
    camera_setup: str | None = None,
) -> nn.Module:
    """
    Load a pretrained model component from an experiment checkpoint.

    Args:
        experiment_spec: Tuple of (experiment_name, checkpoint_iteration)
        component_type: Either "assignment" or "regressor"
        logs_root: Root directory containing experiment logs
        device: Device to load the model onto

    Returns:
        Initialized model with loaded weights
    """
    exp_name, iteration = experiment_spec
    logs_root = Path(logs_root)

    # Determine experiment directory based on component type
    if component_type == "assignment":
        exp_dir = logs_root / "Assignment" / exp_name
    elif component_type == "regressor":
        exp_dir = logs_root / "Pose" / exp_name
    else:
        raise ValueError(f"Unknown component type: {component_type}")

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Load hydra config
    config_path = exp_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Get the component config
    if component_type == "assignment":
        component_cfg = cfg.model.assignment
    else:
        component_cfg = cfg.model.regressor

    # Null out any ckpt_path to avoid double-loading weights
    # (the saved config may reference a bootstrap checkpoint that no longer exists)
    if OmegaConf.is_config(component_cfg) and "ckpt_path" in component_cfg:
        component_cfg = component_cfg.copy()
        component_cfg.ckpt_path = None

    # Override camera_setup throughout the config (e.g., to benchmark with
    # a different number of views using the same pretrained weights).
    if camera_setup is not None:
        _override_camera_setup(component_cfg, camera_setup)

    # Instantiate the model
    model = instantiate(component_cfg)

    # Load checkpoint — try Lightning format first, then legacy
    ckpt_path = None
    for candidate in _checkpoint_candidates(exp_dir, iteration):
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for iteration {iteration} in {exp_dir / 'checkpoints'}"
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Support both Lightning ("state_dict") and legacy ("model") checkpoint formats
    full_state_dict = checkpoint.get("state_dict", checkpoint.get("model", {}))

    # Filter keys for this component (e.g., "assignment_net." or "regressor_net.")
    prefix = f"{component_type}_net."
    component_state_dict = {k.removeprefix(prefix): v for k, v in full_state_dict.items() if k.startswith(prefix)}

    if not component_state_dict:
        raise RuntimeError(
            f"No weights found for '{prefix}' in checkpoint. "
            f"Available prefixes: {set(k.split('.')[0] for k in full_state_dict.keys())}"
        )

    # Remap legacy key names (rayconv was renamed to value_proj)
    component_state_dict = {k.replace(".rayconv.", ".value_proj."): v for k, v in component_state_dict.items()}

    # Load weights
    model.load_state_dict(component_state_dict, strict=True)

    return model
