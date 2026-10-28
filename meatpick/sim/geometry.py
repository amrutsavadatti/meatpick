"""Ideal (undistorted) mapping between the belt frame and the camera image.

Belt frame: x along travel, y across, origin at the belt's left edge (facing travel) under the
camera's first pixel row. The image is that frame rotated 90 degrees clockwise: belt x runs down
the image rows, belt y runs right-to-left across the columns. It is a proper rotation, so a
fat side measured relative to the long-axis normal keeps its sign in both frames.
"""

from __future__ import annotations


def px_per_mm(cfg: dict) -> float:
    return cfg["camera"]["width_px"] / cfg["camera"]["fov_mm"][0]


def fov_along_mm(cfg: dict) -> float:
    return cfg["camera"]["height_px"] / px_per_mm(cfg)


def belt_to_image(cfg: dict, x_mm: float, y_mm: float) -> tuple[float, float]:
    """Belt (x, y) in mm at the camera's reference position -> image (u, v) pixels."""
    ppm = px_per_mm(cfg)
    return (cfg["belt"]["width_mm"] - y_mm) * ppm, x_mm * ppm


def image_to_belt(cfg: dict, u: float, v: float) -> tuple[float, float]:
    ppm = px_per_mm(cfg)
    return v / ppm, cfg["belt"]["width_mm"] - u / ppm


def belt_angle_to_image(theta_deg: float) -> float:
    """Long-axis angle in the belt frame -> angle in the image frame, both in [0, 180)."""
    return (theta_deg + 90.0) % 180.0


def image_angle_to_belt(angle_deg: float) -> float:
    return (angle_deg - 90.0) % 180.0


def fat_side_image_to_belt(angle_img_deg: float, side: int) -> int:
    """Fat side is relative to the normal of the *reduced* angle, which flips when
    ``image_angle_to_belt`` wraps (image angle below 90 degrees)."""
    return side if (angle_img_deg % 180.0) >= 90.0 else -side
