"""Stage B — clean up raw 2D detections.

Two filters:
  1. Per-camera "forbidden polygon" masks (cull boxes that lie fully inside a hand-marked
     dead zone — e.g. dome edges with no usable people).
  2. Drop the smaller of two heavily overlapping boxes when the pose-OKS confirms they are
     the same person.

Finally, convert COCO17 keypoints to the 15-joint panoptic convention.
"""

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm


CONTAINMENT_THRESHOLD = 0.95
OKS_THRESHOLD = 0.65
KP_VIS_THRESH = 0.20


_COCO_SIGMAS = (
    np.array(
        [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
        dtype=np.float32,
    )
    / 10.0
)
_COCO_SIGMAS2 = _COCO_SIGMAS**2


# OKS used as the "same person" test on overlapping boxes.
def oks_between_coco17(a_kps: np.ndarray, b_kps: np.ndarray, ref_area: float, vis_thresh: float) -> float:
    if ref_area <= 0:
        return 0.0
    a_v = a_kps[:, 2] >= vis_thresh
    b_v = b_kps[:, 2] >= vis_thresh
    vis = a_v & b_v
    if not np.any(vis):
        return 0.0
    d2 = np.sum((a_kps[:, :2] - b_kps[:, :2]) ** 2, axis=1).astype(np.float32)
    denom = np.maximum(2.0 * ref_area * _COCO_SIGMAS2, 1e-12)
    return float(np.exp(-d2 / denom)[vis].mean())


def _intersection_area_xywh(b1: np.ndarray, b2: np.ndarray) -> float:
    x1, y1, x2, y2 = b1[0], b1[1], b1[0] + b1[2], b1[1] + b1[3]
    u1, v1, u2, v2 = b2[0], b2[1], b2[0] + b2[2], b2[1] + b2[3]
    w = max(0, min(x2, u2) - max(x1, u1))
    h = max(0, min(y2, v2) - max(y1, v1))
    return w * h


def filter_contained_bboxes_with_pose(
    bboxes: np.ndarray,
    kps_coco17: np.ndarray,
    contain_thresh: float = CONTAINMENT_THRESHOLD,
    oks_thresh: float = OKS_THRESHOLD,
    kp_vis_thresh: float = KP_VIS_THRESH,
) -> np.ndarray:
    """Mask out the smaller of any pair where (a) it lies mostly inside the other and (b) OKS confirms same person."""
    N = bboxes.shape[0]
    if N < 2:
        return np.ones(N, dtype=bool)

    areas = np.where(bboxes[:, 2] * bboxes[:, 3] > 0, bboxes[:, 2] * bboxes[:, 3], 1e-6).astype(np.float32)
    is_kept = np.ones(N, dtype=bool)

    for i in range(N):
        if not is_kept[i]:
            continue
        for j in range(i + 1, N):
            if not is_kept[j]:
                continue
            inter = _intersection_area_xywh(bboxes[i], bboxes[j])
            smaller, _ = (i, j) if areas[i] < areas[j] else (j, i)
            if inter / areas[smaller] <= contain_thresh:
                continue
            ref_area = max(areas[i], areas[j])
            if oks_between_coco17(kps_coco17[i], kps_coco17[j], ref_area, kp_vis_thresh) >= oks_thresh:
                is_kept[smaller] = False
    return is_kept


class FilterBBoxByPosition:
    """Drop a detection box that lies inside a "forbidden" image region.

    Two modes:
      * Per-camera (Panoptic): ``polygons`` is ``{camera: [vertices]}`` and a box is dropped
        only if it lies *fully* inside the camera's polygon.
      * Per-sequence (MM-OR): ``polygons`` is ``{sequence: {camera: [[vertices], ...]}}`` and a
        box is dropped if at least ``overlap_threshold`` of its area falls inside the union of
        that (sequence, camera)'s forbidden polygons.

    Most datasets pass an empty dict (no-op).
    """

    def __init__(
        self,
        polygons: dict | None = None,
        *,
        per_sequence: bool = False,
        overlap_threshold: float | None = None,
    ) -> None:
        polygons = polygons or {}
        self.per_sequence = per_sequence
        self.overlap_threshold = overlap_threshold
        if per_sequence:
            self.polys = {
                seq: {
                    cam: (unary_union([Polygon(p) for p in polys]) if polys else None)
                    for cam, polys in cams.items()
                }
                for seq, cams in polygons.items()
            }
        else:
            self.polys = {k: (Polygon(v) if v else None) for k, v in polygons.items()}

    def _should_drop(self, bbox_xywh: np.ndarray, geom) -> bool:
        if geom is None:
            return False
        x, y, w, h = map(float, bbox_xywh[:4])
        if w <= 0 or h <= 0:
            return False
        rect = box(x, y, x + w, y + h)
        if self.overlap_threshold is None:
            return bool(geom.covers(rect))
        ratio = geom.intersection(rect).area / rect.area if rect.area > 0 else 0.0
        return ratio >= self.overlap_threshold

    def __call__(self, bbox_xywh: np.ndarray, camera_view, sequence_name: str | None = None) -> bool:
        """Return True to KEEP the box, False to drop it."""
        if self.per_sequence:
            geom = self.polys.get(sequence_name, {}).get(camera_view)
        else:
            geom = self.polys.get(camera_view)
        return not self._should_drop(bbox_xywh, geom)


class ConvertCOCOToPanoptic:
    """Map 17-joint COCO keypoints to the 15-joint panoptic skeleton (neck = mean of shoulders, mid-hip = mean of hips)."""

    PANOPTIC = [
        "neck", "nose", "mid-hip",
        "left_shoulder", "left_elbow", "left_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_shoulder", "right_elbow", "right_wrist",
        "right_hip", "right_knee", "right_ankle",
    ]
    COCO17 = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    def __init__(self) -> None:
        self.mapping = [self.COCO17.index(p) if p in self.COCO17 else -1 for p in self.PANOPTIC]
        self.coco_idx = {n: self.COCO17.index(n) for n in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]}
        self.panoptic_idx = {n: self.PANOPTIC.index(n) for n in ["neck", "mid-hip"]}

    def __call__(self, kps: np.ndarray) -> np.ndarray:
        N = kps.shape[0]
        if N == 0:
            return np.zeros((0, 15, 3), dtype=kps.dtype)
        xy, s = kps[:, :, :2], kps[:, :, 2]
        out_xy = np.zeros((N, 15, 2), dtype=xy.dtype)
        out_s = np.zeros((N, 15), dtype=s.dtype)

        # Synthesized joints: neck = mean(shoulders), mid-hip = mean(hips); score = min of pair.
        ls, rs = self.coco_idx["left_shoulder"], self.coco_idx["right_shoulder"]
        lh, rh = self.coco_idx["left_hip"], self.coco_idx["right_hip"]
        out_xy[:, self.panoptic_idx["neck"]] = (xy[:, ls] + xy[:, rs]) / 2.0
        out_s[:, self.panoptic_idx["neck"]] = np.minimum(s[:, ls], s[:, rs])
        out_xy[:, self.panoptic_idx["mid-hip"]] = (xy[:, lh] + xy[:, rh]) / 2.0
        out_s[:, self.panoptic_idx["mid-hip"]] = np.minimum(s[:, lh], s[:, rh])

        # Direct mappings
        for p_idx, c_idx in enumerate(self.mapping):
            if c_idx != -1:
                out_xy[:, p_idx], out_s[:, p_idx] = xy[:, c_idx], s[:, c_idx]
        return np.dstack((out_xy, out_s))


def run_filter(detections: dict, spec) -> dict:
    """Apply position/containment filters and COCO→panoptic keypoint conversion. Returns filtered dict."""
    data = detections
    seq_polygons = getattr(spec, "forbidden_polygons_by_sequence", None)
    if seq_polygons is not None:
        from preparation.utils.datasets.mm_or_polygons import FORBIDDEN_OVERLAP_THRESHOLD

        bbox_filter = FilterBBoxByPosition(
            seq_polygons, per_sequence=True, overlap_threshold=FORBIDDEN_OVERLAP_THRESHOLD,
        )
    else:
        bbox_filter = FilterBBoxByPosition(spec.position_polygons)
    kps_converter = ConvertCOCOToPanoptic()
    cam_keys = [spec.cam_key(c) for c in spec.cameras]
    filtered: dict = {}

    for seq_name, frames in data.items():
        filtered[seq_name] = {}
        for frame_num, frame_data in tqdm(frames.items(), desc=f"Filtering {seq_name}", leave=False):
            filtered[seq_name][frame_num] = {"2D": {}}
            for cam_view in cam_keys:
                view_data = frame_data["2D"].get(cam_view)
                if not view_data or view_data["bbox_xywhs"].shape[0] == 0:
                    filtered[seq_name][frame_num]["2D"][cam_view] = {
                        "keypoints_xys": np.zeros((0, spec.n_joints, 3), dtype=np.float32),
                        "bbox_xywhs": np.zeros((0, 5), dtype=np.float32),
                    }
                    continue

                kps_before, bboxes = view_data["keypoints_xys"], view_data["bbox_xywhs"].copy()
                # Clip boxes to (arbitrarily large) sensible bounds — the actual cap doesn't matter
                # for downstream OKS, only for keeping x/y/w/h non-negative.
                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, None)
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, None)
                bboxes[:, 2] = np.clip(bboxes[:, 2], 1, None)
                bboxes[:, 3] = np.clip(bboxes[:, 3], 1, None)

                # Position filter (per-camera, or per-sequence for MM-OR).
                keep = [i for i, b in enumerate(bboxes) if bbox_filter(b, cam_view, seq_name)]
                kps_pos, bboxes_pos = kps_before[keep], bboxes[keep]

                # Containment+OKS filter.
                mask = (
                    filter_contained_bboxes_with_pose(bboxes_pos, kps_pos)
                    if bboxes_pos.shape[0] > 1
                    else np.ones(bboxes_pos.shape[0], dtype=bool)
                )
                kps_final, bboxes_final = kps_pos[mask], bboxes_pos[mask]

                # COCO 17 → panoptic 15.
                kps_panoptic = kps_converter(kps_final)
                filtered[seq_name][frame_num]["2D"][cam_view] = {
                    "keypoints_xys": kps_panoptic,
                    "bbox_xywhs": bboxes_final,
                }

    return filtered
