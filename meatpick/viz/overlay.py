"""Drawing helpers for detections and truth outlines."""

from __future__ import annotations

import cv2
import numpy as np

PALETTE = [
    (80, 220, 80),
    (60, 180, 255),
    (255, 160, 60),
    (220, 90, 220),
    (90, 230, 230),
    (255, 90, 90),
]


def draw_detections(image: np.ndarray, dets, truth_polygons=None) -> np.ndarray:
    """Tint each detected mask, outline it, and print its confidence. Truth outlines are white."""
    out = image.copy()
    for k, d in enumerate(dets):
        color = PALETTE[k % len(PALETTE)]
        tint = np.zeros_like(out)
        tint[d.mask] = color
        out = cv2.addWeighted(out, 1.0, tint, 0.3, 0)
        cnts, _ = cv2.findContours(
            d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
        ys, xs = np.nonzero(d.mask)
        if len(xs):
            cv2.putText(out, f"{d.confidence:.2f}", (int(xs.mean()) - 18, int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
    for poly in truth_polygons or []:
        cv2.polylines(out, [np.asarray(poly, np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_poses(image: np.ndarray, dets, cal, cfg: dict) -> np.ndarray:
    """Outline, long axis (red), grasp line (yellow), fat side (cream arrow)."""
    out = image.copy()
    for k, d in enumerate(dets):
        color = PALETTE[k % len(PALETTE)]
        cnts, _ = cv2.findContours(
            d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
        a = np.radians(d.angle_deg)
        axis, normal = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        c = np.array([d.x_mm, d.y_mm])

        def px(p):
            u, v = cal.belt_mm_to_px(float(p[0]), float(p[1]))
            return int(round(u)), int(round(v))

        cv2.line(
            out,
            px(c - axis * d.length_mm / 2),
            px(c + axis * d.length_mm / 2),
            (60, 60, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            out,
            px(c - normal * d.width_mm / 2),
            px(c + normal * d.width_mm / 2),
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )
        fat_tip = px(c + normal * d.fat_side * d.width_mm / 2)
        cv2.arrowedLine(out, px(c), fat_tip, (205, 235, 245), 3, cv2.LINE_AA, tipLength=0.25)
        label = (
            f"{d.angle_deg:.0f}deg {d.length_mm:.0f}x{d.width_mm:.0f}mm {d.est_weight_g:.0f}g"
            + (" T" if d.touching else "")
        )
        u0, v0 = px(c)
        cv2.putText(
            out,
            label,
            (u0 - 70, v0 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (u0 - 70, v0 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
