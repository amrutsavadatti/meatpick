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
