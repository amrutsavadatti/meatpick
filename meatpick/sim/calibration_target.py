"""Checkerboard targets for calibration, rendered through the simulated camera.

The board is a plane in 3D, expressed in the *straight-down camera frame*: origin at the camera,
+X to the image right, +Y down the image, +Z toward the belt. The belt plane is Z = d with
d = f_px / ppm. The camera is that pinhole tilted by ``Camera.R_tilt`` and then lens distorted,
so any board pose renders consistently with the belt frames.

Board frame: origin at the outer top-left corner, +X along the top edge, +Y down the left edge,
squares of ``square_mm``. Laid flat on the belt with the default orientation, board +X runs along
belt -y and board +Y along belt +x, so ``flat_origin_mm`` is the belt (x, y) of that corner.
"""

from __future__ import annotations

import cv2
import numpy as np

from meatpick.sim.camera import Camera
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World

PATTERN = (9, 6)  # inner corners (columns, rows); 10 x 7 squares
SQUARE_MM = 25.0
QUIET_MM = 15.0  # white margin around the squares
DARK, LIGHT, PAPER = 40.0, 225.0, 232.0


def inner_corners_board_mm(pattern=PATTERN, square_mm=SQUARE_MM) -> np.ndarray:
    """(N, 2) inner-corner positions in the board frame, row-major, column index fastest."""
    cols, rows = pattern
    j, i = np.mgrid[0:rows, 0:cols]
    return np.stack([(i.ravel() + 1) * square_mm, (j.ravel() + 1) * square_mm], axis=1)


def flat_corners_belt_mm(origin_belt_mm, pattern=PATTERN, square_mm=SQUARE_MM) -> np.ndarray:
    """Where the inner corners sit in the belt frame for a board laid flat at ``origin``."""
    b = inner_corners_board_mm(pattern, square_mm)
    return np.stack([origin_belt_mm[0] + b[:, 1], origin_belt_mm[1] - b[:, 0]], axis=1)


def _belt_to_cam0(cam: Camera, x_mm: float, y_mm: float) -> tuple[float, float]:
    bw = cam.cfg["belt"]["width_mm"]
    return bw - y_mm - cam.W / (2 * cam.ppm), x_mm - cam.H / (2 * cam.ppm)


def belt_distance_mm(cam: Camera) -> float:
    return cam.f_px / cam.ppm


def flat_pose(cam: Camera, origin_belt_mm) -> tuple[np.ndarray, np.ndarray]:
    X, Y = _belt_to_cam0(cam, *origin_belt_mm)
    return np.eye(3), np.array([X, Y, belt_distance_mm(cam)])


def _rodrigues(rvec) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, np.float64))[0]


def board_points_px(cam: Camera, r_b, t_b, pattern=PATTERN, square_mm=SQUARE_MM):
    """Projected inner corners and the four outer corners, in final image pixels."""
    sq = square_mm
    nx, ny = pattern[0] + 1, pattern[1] + 1
    flat = inner_corners_board_mm(pattern, sq)
    inner = np.column_stack([flat, np.zeros(len(flat))])
    outer = np.array([[0, 0, 0], [nx * sq, 0, 0], [nx * sq, ny * sq, 0], [0, ny * sq, 0]], float)
    to_cam0 = lambda p: (np.asarray(r_b) @ p.T).T + t_b  # noqa: E731
    return cam.project_points_3d(to_cam0(inner)), cam.project_points_3d(to_cam0(outer))


def sample_tilted_pose(cam: Camera, rng: np.random.Generator, pattern=PATTERN, square_mm=SQUARE_MM):
    """A random board pose whose whole board lies inside the image."""
    d = belt_distance_mm(cam)
    nx, ny = pattern[0] + 1, pattern[1] + 1
    center_b = np.array([nx * square_mm / 2, ny * square_mm / 2, 0.0])
    for _ in range(500):
        az = rng.uniform(0, 2 * np.pi)
        tilt = np.radians(rng.uniform(8, 40))
        axis = np.array([np.cos(az), np.sin(az), 0.0]) * tilt
        yaw = np.radians(rng.uniform(-25, 25))
        r_b = _rodrigues(axis) @ _rodrigues([0, 0, yaw])
        center = np.array(
            [rng.uniform(-110, 110), rng.uniform(-45, 45), d * rng.uniform(0.82, 1.15)]
        )
        t_b = center - r_b @ center_b
        inner, outer = board_points_px(cam, r_b, t_b, pattern, square_mm)
        pts = np.vstack([inner, outer])
        if (
            pts[:, 0].min() > 25
            and pts[:, 0].max() < cam.W - 25
            and pts[:, 1].min() > 25
            and pts[:, 1].max() < cam.H - 25
            and (r_b @ [0, 0, 1])[2] > 0.5  # facing the camera
        ):
            return r_b, t_b
    raise RuntimeError("could not place a tilted board inside the frame")


def _filtered_square_wave(s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Box-filtered +1/-1 square wave of period 2: antialiases the checker edges."""
    tri = lambda x: 1.0 - np.abs((x % 2.0) - 1.0)  # noqa: E731
    w = np.maximum(w, 1e-4)
    return (tri(s + w / 2) - tri(s - w / 2)) / w


def _board_overlay(cam: Camera, r_b, t_b, pattern, square_mm):
    """Return ``final_overlay(img)`` painting the board into a warped float BGR image."""
    h_b = cam.K @ cam.R_tilt @ np.column_stack([r_b[:, 0], r_b[:, 1], t_b])
    pre = np.concatenate([cam.pre_grid, np.ones(cam.pre_grid.shape[:2] + (1,))], axis=2)
    pts = pre @ np.linalg.inv(h_b).T
    xb, yb = pts[..., 0] / pts[..., 2], pts[..., 1] / pts[..., 2]
    # pixel footprint in board units, for edge antialiasing
    wx = (np.abs(np.gradient(xb, axis=1)) + np.abs(np.gradient(xb, axis=0))) / square_mm
    wy = (np.abs(np.gradient(yb, axis=1)) + np.abs(np.gradient(yb, axis=0))) / square_mm
    checker = _filtered_square_wave(xb / square_mm, wx) * _filtered_square_wave(yb / square_mm, wy)
    nx, ny = pattern[0] + 1, pattern[1] + 1
    on_squares = (xb >= 0) & (xb <= nx * square_mm) & (yb >= 0) & (yb <= ny * square_mm)
    on_paper = (
        (xb >= -QUIET_MM) & (xb <= nx * square_mm + QUIET_MM)
        & (yb >= -QUIET_MM) & (yb <= ny * square_mm + QUIET_MM)
        & (pts[..., 2] > 0)
    )  # fmt: skip
    mid = 0.5 * (DARK + LIGHT)
    gray = np.where(on_squares, mid - 0.5 * (LIGHT - DARK) * checker, PAPER).astype(np.float32)

    def overlay(img: np.ndarray) -> np.ndarray:
        a = on_paper[..., None]
        return np.where(a, gray[..., None], img).astype(np.float32)

    return overlay


def blank_world(cfg: dict, rng: np.random.Generator) -> World:
    """An empty, stationary belt: calibration shots have no meat on them."""
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    world.pieces = []
    world.belt.speed_mps = world.belt.target_mps = 0.0
    return world


def render_board(cam: Camera, world: World, r_b, t_b, pattern=PATTERN, square_mm=SQUARE_MM):
    frame, _ = cam.render(world, final_overlay=_board_overlay(cam, r_b, t_b, pattern, square_mm))
    return frame.image


# Where the operator lays the board flat. Fit placements spread over the field of view; the
# validation placements are different spots that the calibration never saw.
FIT_ORIGINS_MM = [(15.0, 285.0), (15.0, 585.0), (145.0, 285.0), (145.0, 585.0)]
VALIDATION_ORIGINS_MM = [(80.0, 435.0), (150.0, 345.0), (30.0, 500.0)]


def render_calibration_set(cam: Camera, cfg: dict, rng: np.random.Generator, n_tilted: int = 14):
    """Images a calibration session would capture.

    Returns (tilted_images, [(flat_image, origin_belt_mm)...], [(validation_image, origin)...]).
    """
    world = blank_world(cfg, rng)
    tilted = []
    for _ in range(n_tilted):
        r_b, t_b = sample_tilted_pose(cam, rng)
        tilted.append(render_board(cam, world, r_b, t_b))
    flat = [(render_board(cam, world, *flat_pose(cam, o)), o) for o in FIT_ORIGINS_MM]
    held_out = [(render_board(cam, world, *flat_pose(cam, o)), o) for o in VALIDATION_ORIGINS_MM]
    return tilted, flat, held_out
