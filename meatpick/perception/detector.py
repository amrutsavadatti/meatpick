"""What every detector returns, so the classical and YOLO paths are interchangeable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from meatpick.types import Frame


@dataclass
class MaskInstance:
    """One detected piece: its visible pixels at the frame's full resolution."""

    mask: np.ndarray  # HxW bool
    confidence: float  # 0-1
    touches_border: bool = False  # cut off by the frame edge, so only partly visible


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[MaskInstance]: ...
