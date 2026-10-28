"""Pose accuracy against truth: centre, angle, fat side, size, weight."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from meatpick.eval.matching import frame_stems, load_truth_frame, match_masks
from meatpick.types import Frame


def angle_error_deg(a: float, b: float) -> float:
    """Smallest difference between two long-axis angles (they repeat every 180 degrees)."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


def evaluate_pose(
    perceiver, root: str | Path, min_visibility: float = 0.95, limit: int | None = None
) -> dict:
    """Score pose on pieces that are almost fully visible, so the truth pose is well defined."""
    centre, angle, size_l, size_w, weight, fat = [], [], [], [], [], []
    touching_fat = []
    for stem in frame_stems(root)[: limit or None]:
        tf = load_truth_frame(root, stem)
        dets = perceiver.perceive(Frame(tf.image, tf.record["encoder"], tf.record["t_s"]))
        pieces = tf.record["pieces"]
        m = match_masks([d.mask for d in dets], tf.instance_map, pieces)
        truth = {p["id"]: p for p in pieces}
        for di, pid in m.matched:
            t, d = truth[pid], dets[di]
            if t["visibility"] < min_visibility:
                continue
            fat_ok = d.fat_side == t["fat_side"]
            (touching_fat if t["touching"] else fat).append(fat_ok)
            if t["touching"]:
                continue  # a merged or clipped mask has no clean pose to compare
            centre.append(np.hypot(d.x_mm - t["x_mm"], d.y_mm - t["y_mm"]))
            angle.append(angle_error_deg(d.angle_deg, t["angle_deg"]))
            size_l.append(d.length_mm - t["length_mm"])
            size_w.append(d.width_mm - t["width_mm"])
            weight.append((d.est_weight_g - t["weight_g"]) / t["weight_g"])
    q = lambda x, p: round(float(np.percentile(x, p)), 3) if len(x) else float("nan")  # noqa: E731
    return {
        "pieces_scored": len(centre),
        "center_mm_median": q(centre, 50),
        "center_mm_p95": q(centre, 95),
        "angle_deg_median": q(angle, 50),
        "angle_deg_p95": q(angle, 95),
        "fat_side_accuracy_isolated": round(float(np.mean(fat)), 4) if fat else float("nan"),
        "fat_side_accuracy_touching": round(float(np.mean(touching_fat)), 4)
        if touching_fat
        else float("nan"),
        "length_mm_abs_median": q(np.abs(size_l), 50),
        "width_mm_abs_median": q(np.abs(size_w), 50),
        "weight_rel_abs_median": q(np.abs(weight), 50),
        "weight_rel_abs_p95": q(np.abs(weight), 95),
    }
