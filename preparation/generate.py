"""Multi-dataset pseudo-label generation (train split) via the COMPOSE library.

Detection (RT-DETR + ViTPose++) and 2D filtering (position polygons, OKS dedup,
COCO-17 -> panoptic-15) stay local: COMPOSE has no detector and consumes a ready
2D-detections pickle. Association, triangulation, the pseudo-label output, and the
multi-view visualization are delegated to COMPOSE so this repo no longer carries
its own copy of that pipeline.

Output is the same pseudo-label pickle the pose training reads (the filename always
uses the `panoptic_` prefix, regardless of dataset — see RunPaths.pseudo_labels_path):
    data/preparation/<dataset>/<timestamp>_train/panoptic_<timestamp>_pseudo_labels.pkl
Sampled multi-view grid visualizations (skeletons coloured by associated id) are
written to preparation/output/<dataset>/<timestamp>_train/.

Usage:
    python preparation/generate.py --dataset panoptic
    python preparation/generate.py --dataset mm_or --interval 3
    # Re-render visualizations into an existing run without re-detecting:
    python preparation/generate.py --dataset panoptic --viz-only \
        --timestamp 2026_06_03_15_47 \
        --detections-pkl data/preparation/panoptic/2026_06_03_15_47_train/detections.pkl
"""

import argparse
import pickle
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compose import PipelineConfig, build_cues, get_image_path, load_dataset, process_frame
from compose.camera import world_3d_to_img_2d
from compose.filtering import AnatomicalFilter
from compose.visualization import add_frame_info_to_image, save_image, visualize_frame

from preparation.utils.config import PrepareConfig, RunPaths
from preparation.utils.datasets import get_spec
from preparation.utils.detection import run_detection
from preparation.utils.filter import run_filter

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def _save_visualization(
    data_batch, prediction, ilp_matches, hungarian_matches, weights_filtered,
    config, device, seq_name, frame_num, out_dir, patient_dets=None,
):
    """Render and save a multi-view grid (skeletons coloured by associated id) for one frame."""
    mv_images = np.array([
        cv2.imread(str(get_image_path(config.dataset, config.dataset_path, seq_name, cam, frame_num)))
        for cam in config.cameras
    ])
    cam_params_vec = data_batch["cam_params_vec"].to(device)
    world_xyz = torch.from_numpy(prediction[:, :, :3]).to(device)
    proj_uv, mask = world_3d_to_img_2d(world_xyz, cam_params_vec.unsqueeze(1))
    proj_uv[~mask.squeeze(-1)] = -1.0
    proj_uv_np = proj_uv[0, 0].detach().cpu().numpy()
    valid_3d_np = prediction[:, :, -1]
    all_keypoints_np = data_batch["keypoints_xys"].squeeze(0).cpu().numpy()
    if patient_dets:  # drop the patient's 2D detections so they are not drawn at all
        for v, n in patient_dets:
            all_keypoints_np[v, n, :, 2] = 0.0

    combined = visualize_frame(
        mv_images, all_keypoints_np, ilp_matches, hungarian_matches,
        proj_uv_np, valid_3d_np, weights_filtered, prediction.shape[0], config,
    )
    add_frame_info_to_image(
        combined, frame_num, None, font_scale=config.font_scale, thickness=config.font_thickness,
    )
    save_image(combined, out_dir / f"{seq_name}_{frame_num:06d}.jpg", config.jpeg_quality)


_ANKLE_JOINTS = (8, 14)  # panoptic-15 left/right ankle


def _patient_rows(prediction: np.ndarray, foot_z_threshold: float) -> set[int]:
    """Row indices of triangulated persons with a valid ankle above the height threshold (mm)."""
    rows: set[int] = set()
    for i in range(prediction.shape[0]):
        for a in _ANKLE_JOINTS:
            if prediction[i, a, 3] > 0 and prediction[i, a, 2] > foot_z_threshold:
                rows.add(i)
                break
    return rows


def _drop_patients(frame_output: dict, patient_rows: set[int]) -> None:
    """Remove patient persons (3D row == person id) from the 3D and 2D output dicts, in place."""
    tri = frame_output["3D"]["triangulated_keypoints_xyzs"]
    pid3d = frame_output["3D"]["person_ids"]
    keep = np.array([i not in patient_rows for i in range(tri.shape[0])], dtype=bool)
    patient_pids = [int(pid3d[i]) for i in patient_rows]
    frame_output["3D"]["triangulated_keypoints_xyzs"] = tri[keep]
    frame_output["3D"]["person_ids"] = pid3d[keep]
    for cam_data in frame_output["2D"].values():
        mask = ~np.isin(cam_data["person_ids"], patient_pids)
        cam_data["keypoints_xys"] = cam_data["keypoints_xys"][mask]
        cam_data["bbox_xywhs"] = cam_data["bbox_xywhs"][mask]
        cam_data["person_ids"] = cam_data["person_ids"][mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["panoptic", "shelf", "campus", "mm_or"], default="panoptic",
        help="Dataset to process (default: panoptic).",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Frame stride (defaults to the spec — panoptic/mm_or=3, shelf/campus=1).",
    )
    parser.add_argument(
        "--sequences", nargs="+", default=None,
        help="Override the train sequences (default: the dataset spec's full train list).",
    )
    parser.add_argument(
        "--detections-pkl", type=Path, default=None,
        help="Skip detection+filtering and reuse this pre-computed filtered-detections pkl "
             "(must already be keyed the way COMPOSE indexes views).",
    )
    parser.add_argument(
        "--visualize", action=argparse.BooleanOptionalAction, default=True,
        help="Save sampled multi-view grid visualizations to the run's viz dir.",
    )
    parser.add_argument(
        "--viz-only", action="store_true",
        help="Only (re)render visualizations into an existing run; do not write the "
             "pseudo-label pkl. Requires --detections-pkl and --timestamp.",
    )
    parser.add_argument(
        "--timestamp", type=str, default=None,
        help="Use this run timestamp instead of 'now' (to target an existing run dir).",
    )
    args = parser.parse_args()

    if args.viz_only and (args.detections_pkl is None or args.timestamp is None):
        parser.error("--viz-only requires both --detections-pkl and --timestamp")

    spec = get_spec(args.dataset)
    if args.sequences is not None:
        spec.sequences = list(args.sequences)
    interval = args.interval if args.interval is not None else spec.default_interval
    sequences = list(spec.sequences)

    paths = RunPaths(spec.name, spec.output_labels_base, spec.output_viz_base, timestamp=args.timestamp)
    paths.save_config(PrepareConfig(interval=interval))

    print(f"Dataset       : {spec.name}")
    print(f"Sequences     : {len(sequences)}  {sequences}")
    print(f"Cameras       : {spec.cameras}")
    print(f"Interval      : {interval}")
    print(f"Visualize     : {args.visualize}{' (viz-only)' if args.viz_only else ''}")
    if not args.viz_only:
        print(f"Pseudo-labels -> {paths.pseudo_labels_path}")
    print(f"Visualizations -> {paths.viz_dir}")

    # 1. Detection + 2D filtering (retained front-end; COMPOSE has no detector).
    if args.detections_pkl is not None:
        print(f"Reusing detections : {args.detections_pkl}")
        with open(args.detections_pkl, "rb") as f:
            filtered = pickle.load(f)
    else:
        detections = run_detection(spec, interval=interval)
        filtered = run_filter(detections, spec)

    # Persist detections in COMPOSE's input schema (unless we're only re-rendering viz).
    if args.viz_only:
        detections_path = str(args.detections_pkl)
    else:
        detections_path = str(paths.labels_dir / "detections.pkl")
        with open(detections_path, "wb") as f:
            pickle.dump(filtered, f)

    # 2. Association + triangulation (+ optional visualization) via COMPOSE.
    config = PipelineConfig.from_yaml(
        EXPERIMENTS_DIR / f"{spec.name}_train.yaml",
        split="train",
        overrides={
            "detections_path": detections_path,
            "sequences": sequences,
            "interval": interval,
            "save_video": False,
        },
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(config)
    config.root_id = dataset.root_id
    cues = build_cues(config.cues, num_views=config.V, device=device)
    anatomical_filter = AnatomicalFilter(
        config.dataset, convention_2d=dataset.convention_2d, convert=dataset.convert,
    ).to(device)

    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)

    foot_z_threshold = spec.patient_foot_z_threshold_mm

    pose_data: dict[str, dict[int, dict]] = {}
    for idx, data_batch in enumerate(tqdm(loader, desc=f"Associating {spec.name}")):
        seq_name = data_batch["sequence"][0]
        frame = data_batch["frame"][0]
        frame_num = int(frame.item()) if torch.is_tensor(frame) else int(frame)
        prediction, frame_output, ilp_matches, hungarian_matches, weights_filtered = process_frame(
            data_batch, cues, anatomical_filter, config, device,
        )

        # Drop the patient (elevated 3D feet) from both 3D and 2D, for datasets that set the threshold.
        patients = (
            _patient_rows(prediction, foot_z_threshold)
            if foot_z_threshold is not None and prediction.shape[0] > 0
            else set()
        )
        if patients:
            _drop_patients(frame_output, patients)
        pose_data.setdefault(seq_name, {})[frame_num] = frame_output

        if args.visualize and prediction.shape[0] > 0 and (idx % config.save_image_interval) == 0:
            pred_viz, ilp_viz, hungarian_viz = prediction, ilp_matches, hungarian_matches
            patient_dets: set = set()
            if patients:  # exclude the patient from the rendered frame too (3D skeleton + its 2D detections)
                pred_viz = prediction.copy()
                for i in patients:
                    pred_viz[i, :, 3] = 0.0
                ilp_viz = {k: v for k, v in ilp_matches.items() if v not in patients}
                hungarian_viz = {k: v for k, v in hungarian_matches.items() if v not in patients}
                patient_dets = {
                    (v, n)
                    for matches in (ilp_matches, hungarian_matches)
                    for (v, n), p in matches.items()
                    if p in patients
                }
            _save_visualization(
                data_batch, pred_viz, ilp_viz, hungarian_viz, weights_filtered,
                config, device, seq_name, frame_num, paths.viz_dir, patient_dets,
            )

    if not args.viz_only:
        with paths.pseudo_labels_path.open("wb") as f:
            pickle.dump(pose_data, f)
        print(f"\nDone. Pseudo-labels: {paths.pseudo_labels_path}")
    else:
        print(f"\nDone. Visualizations: {paths.viz_dir}")


if __name__ == "__main__":
    main()
