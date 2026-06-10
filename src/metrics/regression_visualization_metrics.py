import numpy as np
import torch
from torchmetrics import Metric

from src.utils.visualization.plot import plot_keypoints_xyz


class RegressionVisualizationMetrics(Metric):
    def __init__(
        self,
        num_frames: int,
        space_size: list,
        dist_sync_on_step=True,
        compute_on_cpu=False,
        sync_on_compute=True,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
            compute_on_cpu=compute_on_cpu,
        )
        self.name = "Regression Visualization"
        self.num_frames = num_frames
        self.space_size = space_size

        self.add_state("regression_pred_xyz", default=[])
        self.add_state("regression_conf", default=[])
        self.add_state("regression_gt_xyzs", default=[])

    def compute(self, **kwargs):
        B = len(self.regression_pred_xyz)

        random_idx = np.random.choice(B, self.num_frames, replace=B < self.num_frames)

        images = []
        for i in random_idx:
            regression_pred_xyz = self.regression_pred_xyz[i].cpu().numpy()
            regression_conf = self.regression_conf[i].cpu().numpy()
            regression_gt_xyzs = self.regression_gt_xyzs[i].cpu().numpy()

            # List of RGB Images
            plots = plot_keypoints_xyz(
                regression_pred_xyz,
                regression_gt_xyzs,
                regression_conf,
            )

            images.append(plots)

        mr_metrics = {"images": images}
        return mr_metrics

    def update(self, regression_pred_xyz: torch.Tensor, confidence: torch.Tensor, regression_gt_xyzs: torch.Tensor):
        self.regression_pred_xyz.extend(regression_pred_xyz)
        self.regression_conf.extend(confidence)
        self.regression_gt_xyzs.extend(regression_gt_xyzs)
