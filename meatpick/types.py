"""Interfaces shared between modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:  # the ONLY thing perception receives
    image: np.ndarray  # HxWx3 uint8, BGR
    encoder: int  # belt encoder count at capture (1 count = 0.1 mm)
    t_s: float


@dataclass
class Detection:
    mask: np.ndarray  # HxW bool
    x_mm: float
    y_mm: float  # center in belt frame
    angle_deg: float  # long axis, [0, 180)
    fat_side: int  # +1 or -1 relative to long axis normal
    length_mm: float
    width_mm: float
    est_weight_g: float
    touching: bool
    confidence: float


@dataclass
class Track:
    id: int
    x_mm: float
    y_mm: float  # at reference encoder count
    ref_encoder: int
    angle_deg: float
    fat_side: int
    confidence: float
    deadline_s: float  # when it leaves the reach zone
    status: str  # tentative | confirmed | assigned | picked | missed
    # Extras beyond the PRD's fields, all defaulted, that the scheduler (#12) needs:
    touching: bool = False  # another piece within gripper clearance
    length_mm: float = 0.0
    width_mm: float = 0.0
    est_weight_g: float = 0.0
    hits: int = 0  # detections matched to this track
    last_obs_encoder: int = 0  # encoder count of the last matched detection


@dataclass
class PickCommand:
    track_id: int
    pick_x_mm: float
    pick_y_mm: float  # robot frame, at grip time
    gripper_yaw_deg: float  # long axis + 90
    grip_at_encoder: int
    grip_at_t_s: float
    place: str = "BIN"
