import logging
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

torch.set_float32_matmul_precision("high")

warnings.filterwarnings("ignore", category=UserWarning, module="mmcv")
warnings.filterwarnings("ignore", message=".*tensorboardX.*")
warnings.filterwarnings("ignore", message=".*litlogger.*")
warnings.filterwarnings("ignore", message=".*litmodels.*")
warnings.filterwarnings("ignore", message=".*custom batch sampler.*")

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models.dispose_module import _format_metrics_table
from src.utils.paramUtil import convert_panoptic_to_campus
from src.utils.resume import resume_experiment
from src.utils.visualization.eval_visualizer import run_visualization

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------------

def _load_checkpoint(cfg, model):
    """Load checkpoint into model. Returns ckpt_path for Lightning (.ckpt) or None for legacy (.pt)."""
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        return None

    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".ckpt":
        return str(ckpt_path)

    # Legacy .pt format — load manually, return None so trainer doesn't try to load again
    log.info(f"Loading legacy checkpoint: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", {}))
    model.load_state_dict(state_dict, strict=False)
    return None


# ------------------------------------------------------------------
# Prediction saving
# ------------------------------------------------------------------

def _save_predictions(predictions: list[dict], output_dir: Path, datamodule) -> None:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    T = datamodule.test_dataloader().dataset.num_temporal
    skeleton_type = datamodule.test_dataloader().dataset.skeleton_type

    for batch_dict in predictions:
        refined = batch_dict.get("refined_poses_xyz")
        roots = batch_dict.get("assignment_xyzs")
        B = next(iter(batch_dict.values())).shape[0] if batch_dict else 0

        for s in range(B):
            if refined is not None:
                poses_xyz = refined[s, 0, T // 2, -1]
                pred_conf = roots[s, 0, T // 2, ..., -1] if roots is not None else np.ones(poses_xyz.shape[0])
            elif roots is not None:
                roots_sample = roots[s, 0, T // 2]
                poses_xyz = roots_sample[..., :3][..., None, :].repeat(15, -2)
                pred_conf = roots_sample[..., -1]
            else:
                continue

            if skeleton_type in ["shelf", "campus"] and poses_xyz.shape[1] == 15:
                poses_xyz = convert_panoptic_to_campus(torch.from_numpy(poses_xyz)).numpy()

            # Parse frame info
            img_paths = batch_dict["img_paths"]
            sequence = batch_dict["sequence"][s]
            if isinstance(sequence, np.ndarray):
                sequence = str(sequence)
            try:
                path = Path(img_paths[s][0][T // 2][0])
            except (TypeError, KeyError):
                path = Path(img_paths[s, 0, T // 2, 0])
            try:
                frame_num = int(path.stem.split("_")[-1])
            except ValueError:
                frame_num = int(path.stem.split("-")[-1])

            # Save
            valid_mask = pred_conf > 0
            n_valid = int(valid_mask.sum())
            if n_valid > 0:
                valid_poses = poses_xyz[valid_mask]
                J = valid_poses.shape[1]
                output = np.zeros((n_valid, J, 4), dtype=np.float32)
                output[:, :, :3] = valid_poses[..., :3]
                output[:, :, 3] = pred_conf[valid_mask][:, np.newaxis]
            else:
                J = poses_xyz.shape[1] if len(poses_xyz.shape) > 1 else 15
                output = np.zeros((0, J, 4), dtype=np.float32)

            seq_dir = predictions_dir / sequence
            seq_dir.mkdir(parents=True, exist_ok=True)
            with open(seq_dir / f"{frame_num:06d}.pkl", "wb") as f:
                pickle.dump(output, f)

    log.info(f"Saved predictions to {predictions_dir}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    cfg = resume_experiment(cfg)

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating model")
    model = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.data)

    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    ckpt_path = _load_checkpoint(cfg, model)
    output_dir = Path(cfg.paths.output_dir)

    # Single pass: predictions + metrics
    predictions_raw = trainer.predict(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    predictions = [
        {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        for d in predictions_raw
    ]
    _save_predictions(predictions, output_dir, datamodule)

    # Log metrics accumulated during predict — same box-drawn table format as
    # training-time validation, routed through the Python logger so it lands in
    # both stdout and the Hydra `.log` file under the run dir.
    metrics = model.compute_metrics()
    if metrics:
        log.info(f"Results\n{_format_metrics_table(metrics)}")
    model.reset_metrics()

    # Visualization
    if cfg.get("visualize", False):
        log.info("Generating visualization")
        collated = defaultdict(list)
        for d in predictions:
            for key, value in d.items():
                collated[key].append(value)
        info_dict = {key: np.concatenate(arrs, axis=0) for key, arrs in collated.items()}
        run_visualization(info_dict, output_dir, datamodule)


if __name__ == "__main__":
    main()
