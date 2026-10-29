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


BORDER_TOL_PX = 8  # YOLO masks stop a few pixels short of the frame edge, so "touching" is fuzzy


def touches_border(mask: np.ndarray, tol_px: int = BORDER_TOL_PX) -> bool:
    """True if the mask comes within ``tol_px`` of any frame edge (the piece is cut off)."""
    t = tol_px
    return bool(mask[:t].any() or mask[-t:].any() or mask[:, :t].any() or mask[:, -t:].any())


def touches_bottom(mask: np.ndarray, tol_px: int = BORDER_TOL_PX) -> bool:
    return bool(mask[-tol_px:].any())
