"""Pick a detector: YOLO when the weights are there, otherwise the classical one."""

from __future__ import annotations

import logging
from pathlib import Path

from meatpick.perception.calibration import Calibration
from meatpick.perception.classical import ClassicalDetector
from meatpick.perception.detector import Detector
from meatpick.perception.pose import poses_for
from meatpick.types import Detection, Frame

log = logging.getLogger("meatpick.perception")


def build_detector(cfg: dict, cal: Calibration, prefer: str | None = None) -> tuple[Detector, str]:
    """Returns (detector, description). Never raises for a missing model: work is not blocked."""
    prefer = prefer or cfg["perception"]["detector"]
    if prefer == "yolo":
        path = Path(cfg["perception"]["model_path"])
        if not path.exists():
            reason = f"{path} not found"
        else:
            try:
                from meatpick.perception.yolo import YoloDetector

                return YoloDetector(cfg, str(path)), f"yolo ({path})"
            except ImportError as e:
                reason = f"ultralytics not installed ({e})"
        log.warning("falling back to the classical detector: %s", reason)
        return ClassicalDetector(cfg, cal.px_per_mm()), f"classical (fallback: {reason})"
    return ClassicalDetector(cfg, cal.px_per_mm()), "classical"


class Perceiver:
    """Frame in, posed detections out. The only thing the rest of the loop calls."""

    def __init__(self, cfg: dict, cal: Calibration, prefer: str | None = None):
        self.cfg, self.cal = cfg, cal
        self.detector, self.description = build_detector(cfg, cal, prefer)

    def perceive(self, frame: Frame) -> list[Detection]:
        return poses_for(self.detector.detect(frame), frame.image, self.cal, self.cfg)
