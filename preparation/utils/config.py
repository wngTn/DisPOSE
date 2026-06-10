"""Algorithmic config + run-directory bookkeeping. Per-dataset I/O lives in `datasets/`."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


def _get_timestamp() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M")


@dataclass
class PrepareConfig:
    """Algorithmic hyperparameters shared across all 4 datasets.

    Dataset-specific values (sequences, cameras, paths) come from `DatasetSpec`.
    """

    interval: int = 3                                # frame interval (CLI flag)

    # Hypergraph
    v_max: int = 5                                   # max views
    n_max: int = 10                                  # max persons per view

    # Confidence thresholds
    hedge_conf_thresh: float = 0.4
    single_hedge_score: float = 1.0
    root_min_conf: float = 0.2
    triangulation_threshold: float = 0.3
    confidence_threshold: float = 0.3

    # ILP penalty
    lam: float = 2.5

    # Geometric cue / anatomical-filter / Hungarian-rematch hyperparameters.
    pixel_threshold: float = 64.0**2                 # px² (matches COMPOSE_Private's geometric cue)
    sigma_threshold: float = 3.5
    min_conf: float = 0.3
    hungarian_cost_threshold: float = 96.0

    # Visualization
    visualize_every: int = 256
    jpeg_quality: int = 75
    line_thickness: int = 6
    keypoint_radius: int = 8
    font_scale: float = 0.8

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


class RunPaths:
    """Output directory layout for a single pipeline run.

    The final pseudo-labels pkl lives under `data/preparation/<dataset>/<ts>_train/` so
    the existing pose training configs continue to find it. Per-frame visualizations live
    under `preparation/output/<dataset>/<ts>_train/`.
    """

    def __init__(
        self,
        dataset: str,
        labels_base_dir: Path | None = None,
        viz_base_dir: Path | None = None,
        timestamp: str | None = None,
    ):
        self.dataset = dataset
        self.timestamp = timestamp or _get_timestamp()
        labels_base = labels_base_dir if labels_base_dir is not None else Path("./data/preparation") / dataset
        viz_base = viz_base_dir if viz_base_dir is not None else Path("./preparation/output") / dataset
        self.labels_dir = labels_base / f"{self.timestamp}_train"
        self.viz_dir = viz_base / f"{self.timestamp}_train"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    @property
    def pseudo_labels_path(self) -> Path:
        # The `panoptic_` prefix is hardcoded for all datasets: the pose dataloader
        # globs for this exact filename regardless of which dataset produced it.
        return self.labels_dir / f"panoptic_{self.timestamp}_pseudo_labels.pkl"

    @property
    def config_path(self) -> Path:
        return self.viz_dir / "config.json"

    def save_config(self, config: PrepareConfig) -> None:
        d = config.to_dict()
        d["dataset"] = self.dataset
        with self.config_path.open("w") as f:
            json.dump(d, f, indent=2)
