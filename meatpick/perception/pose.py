"""Pose from a mask, in belt millimetres: centre, long axis, fat side, size, weight.

Everything is measured after mapping mask pixels through the calibration, so angle and size are
metric even though the lens distorts and tilts the image. Belt frame: x along travel, y across.
The long-axis angle is in [0, 180); ``fat_side`` is +1 when the fat lies on the +normal side,
with normal = (-sin a, cos a), the same convention as the simulator's truth.
"""

from __future__ import annotations

import cv2
import numpy as np

from meatpick.perception.calibration import Calibration
from meatpick.perception.detector import MaskInstance
from meatpick.types import Detection


def fat_side_from_bands(paleness: np.ndarray, s: np.ndarray, band_mm: tuple[float, float]) -> int:
    """+1 if the +normal long edge is paler (fat is pale, lean is pink), else -1."""
    d0, d1 = band_mm
    lo = paleness[(s >= s.min() + d0) & (s <= s.min() + d1)]
    hi = paleness[(s <= s.max() - d0) & (s >= s.max() - d1)]
    if lo.size == 0 or hi.size == 0:  # very thin piece: fall back to the outer thirds
        third = (s.max() - s.min()) / 3
        lo, hi = paleness[s <= s.min() + third], paleness[s >= s.max() - third]
    return 1 if np.median(hi) > np.median(lo) else -1


def extract_pose(
    mask: np.ndarray, image: np.ndarray, cal: Calibration, cfg: dict, confidence: float = 1.0
) -> Detection:
    p = cfg["perception"]["pose"]
    prod = cfg["product"]
    ys, xs = np.nonzero(mask)
    stride = max(1, len(xs) // p["max_points"])
    u, v = xs[::stride].astype(np.float64), ys[::stride].astype(np.float64)
    bx, by = cal.px_to_belt_mm(u, v)
    pts = np.stack([bx, by], axis=1)
    centre = pts.mean(axis=0)
    d = pts - centre

    evals, evecs = np.linalg.eigh(np.cov(d.T))
    ax = evecs[:, np.argmax(evals)]
    angle = float(np.degrees(np.arctan2(ax[1], ax[0])) % 180.0)
    a = np.radians(angle)
    axis, normal = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    t, s = d @ axis, d @ normal
    length, width = float(t.max() - t.min()), float(s.max() - s.min())

    # fat side: compare how pale the two long edges are, inside the (shaded) very edge
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    lab = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    ly, lx = (v - y0).astype(int), (u - x0).astype(int)
    chroma = np.hypot(lab[ly, lx, 1] - 128.0, lab[ly, lx, 2] - 128.0)
    paleness = lab[ly, lx, 0] - p["chroma_weight"] * chroma
    fat_side = fat_side_from_bands(paleness, s, tuple(p["fat_band_mm"]))

    ppm = cal.px_per_mm(float(xs.mean()), float(ys.mean()))
    area_mm2 = len(xs) / ppm**2
    thickness = float(np.mean(prod["thickness_mm"]))  # assumed: no depth sensing
    weight = area_mm2 * thickness * prod["density_g_per_mm3"]
    return Detection(
        mask=mask,
        x_mm=float(centre[0]),
        y_mm=float(centre[1]),
        angle_deg=angle,
        fat_side=fat_side,
        length_mm=length,
        width_mm=width,
        est_weight_g=float(weight),
        touching=False,
        confidence=float(confidence),
    )


def touching_flags(masks: list[np.ndarray], px_per_mm: float, clearance_mm: float) -> list[bool]:
    """True for every mask that has another mask within ``clearance_mm`` (the gripper's reach)."""
    n = len(masks)
    if n < 2:
        return [False] * n
    scale = 0.25  # distances at quarter resolution are plenty for a 25 mm threshold
    small = [
        cv2.resize(m.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        > 0
        for m in masks
    ]
    dist = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) for m in small]
    limit = clearance_mm * px_per_mm * scale
    flags = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            if small[j].any() and dist[i][small[j]].min() <= limit:
                flags[i] = flags[j] = True
    return flags


def poses_for(
    instances: list[MaskInstance], image: np.ndarray, cal: Calibration, cfg: dict
) -> list[Detection]:
    dets = [extract_pose(m.mask, image, cal, cfg, m.confidence) for m in instances]
    ppm = cal.px_per_mm()
    for det, flag in zip(
        dets,
        touching_flags([d.mask for d in dets], ppm, cfg["robot"]["touching_clearance_mm"]),
        strict=True,
    ):
        det.touching = flag
    return dets
