"""Camera calibration from checkerboard images: lens model plus image-to-belt homography.

Inputs are images only, plus what an operator knows: the board's inner-corner pattern, the
square size, and where the board was laid on the belt. Nothing here reads the simulator.

Two steps, as on a real line:
1. Several tilted-board views give the intrinsics and lens distortion (a flat target seen at a
   single tilt cannot fix the focal length).
2. Boards laid flat on the belt at known places give the homography from undistorted pixels
   to belt millimetres.

Belt frame: x along travel, y across. A flat board is described by the belt position of its
outer top-left corner; its +X edge runs along belt -y and its +Y edge along belt +x.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PATTERN = (9, 6)  # inner corners (columns, rows)
SQUARE_MM = 25.0


class CalibrationError(RuntimeError):
    pass


def board_corners_mm(pattern=PATTERN, square_mm=SQUARE_MM) -> np.ndarray:
    """Inner corners in the board frame, row-major with the column index fastest."""
    cols, rows = pattern
    j, i = np.mgrid[0:rows, 0:cols]
    return np.stack([(i.ravel() + 1) * square_mm, (j.ravel() + 1) * square_mm], axis=1)


def flat_board_corners_belt_mm(origin_belt_mm, pattern=PATTERN, square_mm=SQUARE_MM) -> np.ndarray:
    b = board_corners_mm(pattern, square_mm)
    return np.stack([origin_belt_mm[0] + b[:, 1], origin_belt_mm[1] - b[:, 0]], axis=1)


def find_corners(image: np.ndarray, pattern=PATTERN) -> np.ndarray | None:
    """Sub-pixel inner corners in canonical order, or None if the board is not found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            return None
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), term)
    return _canonical_order(corners.reshape(-1, 2).astype(np.float64), pattern)


def _canonical_order(corners: np.ndarray, pattern) -> np.ndarray:
    """Board is shown roughly upright: first row should run right, first column down."""
    cols, rows = pattern
    along_row = corners[cols - 1] - corners[0]
    along_col = corners[(rows - 1) * cols] - corners[0]
    if along_row[0] < 0 and along_col[1] < 0:
        return corners[::-1].copy()  # detected rotated by 180 degrees
    if along_row[0] < 0 or along_col[1] < 0:
        raise CalibrationError("board orientation is ambiguous (mirrored or rotated ~90 degrees)")
    return corners


@dataclass
class Calibration:
    K: np.ndarray  # 3x3 intrinsics
    dist: np.ndarray  # (k1, k2, p1, p2, k3), here k1 and k2 only
    H: np.ndarray  # undistorted pixel -> belt mm
    image_size: tuple[int, int]  # (width, height)
    rms_px: float  # calibrateCamera RMS reprojection error
    n_views: int

    # --- building -------------------------------------------------------------------------
    @classmethod
    def calibrate(
        cls,
        tilted_images: list[np.ndarray],
        flat_views: list[tuple[np.ndarray, tuple[float, float]]],
        pattern=PATTERN,
        square_mm=SQUARE_MM,
    ) -> Calibration:
        obj = np.column_stack(
            [board_corners_mm(pattern, square_mm), np.zeros(pattern[0] * pattern[1])]
        )
        obj = obj.astype(np.float32)
        img_pts, obj_pts, flat_px = [], [], []
        for im in tilted_images:
            c = find_corners(im, pattern)
            if c is not None:
                img_pts.append(c.astype(np.float32))
                obj_pts.append(obj)
        n_tilted = len(img_pts)
        for im, origin in flat_views:
            c = find_corners(im, pattern)
            if c is None:
                raise CalibrationError(f"board not found in the flat view at {origin}")
            img_pts.append(c.astype(np.float32))
            obj_pts.append(obj)
            flat_px.append((c, flat_board_corners_belt_mm(origin, pattern, square_mm)))
        if n_tilted < 6:
            raise CalibrationError(
                f"need at least 6 tilted views with a visible board, got {n_tilted}"
            )
        h, w = tilted_images[0].shape[:2]
        # Radial distortion with two coefficients, no tangential term: the usual model for a
        # machine-vision lens. Principal point and focal length are free.
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
        rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None, flags=flags)
        cal = cls(K=K, dist=dist.ravel(), H=np.eye(3), image_size=(w, h), rms_px=float(rms),
                  n_views=len(img_pts))  # fmt: skip
        px = np.vstack([p for p, _ in flat_px])
        belt = np.vstack([b for _, b in flat_px])
        cal.H, _ = cv2.findHomography(cal._undistort(px), belt, method=0)
        return cal

    # --- mapping --------------------------------------------------------------------------
    def _undistort(self, px: np.ndarray) -> np.ndarray:
        pts = np.asarray(px, np.float64).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, 1e-10)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K, criteria=crit).reshape(-1, 2)

    def px_to_belt_mm(self, u, v):
        """Image pixels (scalars or arrays) -> belt-frame (x_mm, y_mm)."""
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        und = self._undistort(np.stack([u.ravel(), v.ravel()], axis=1))
        belt = cv2.perspectiveTransform(und.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        x, y = belt[:, 0].reshape(u.shape), belt[:, 1].reshape(u.shape)
        return (float(x), float(y)) if x.ndim == 0 else (x, y)

    def belt_mm_to_px(self, x_mm, y_mm):
        """Belt-frame mm -> image pixels, including lens distortion."""
        x, y = np.asarray(x_mm, np.float64), np.asarray(y_mm, np.float64)
        pre = cv2.perspectiveTransform(
            np.stack([x.ravel(), y.ravel()], axis=1).reshape(-1, 1, 2), np.linalg.inv(self.H)
        ).reshape(-1, 2)
        f, c = self.K[0, 0], self.K[:2, 2]
        norm = (pre - c) / f
        r2 = (norm**2).sum(1, keepdims=True)
        k1, k2 = self.dist[0], self.dist[1]
        px = norm * (1 + k1 * r2 + k2 * r2**2) * f + c
        u, v = px[:, 0].reshape(x.shape), px[:, 1].reshape(x.shape)
        return (float(u), float(v)) if u.ndim == 0 else (u, v)

    def px_per_mm(self, u: float | None = None, v: float | None = None) -> float:
        """Local image scale (pixels per belt mm) at a pixel, default the image centre."""
        u = self.image_size[0] / 2 if u is None else u
        v = self.image_size[1] / 2 if v is None else v
        x, y = self.px_to_belt_mm(u, v)
        u1, v1 = self.belt_mm_to_px(x + 1.0, y)
        u2, v2 = self.belt_mm_to_px(x, y + 1.0)
        return float(np.sqrt(np.hypot(u1 - u, v1 - v) * np.hypot(u2 - u, v2 - v)))

    def view_span_x_mm(self) -> tuple[float, float]:
        """Belt x (along travel) at the top and bottom rows of the image, centre column. The
        camera is tilted, so this is not simply [0, height / scale]."""
        u = self.image_size[0] / 2
        x_top, _ = self.px_to_belt_mm(u, 0.0)
        x_bottom, _ = self.px_to_belt_mm(u, float(self.image_size[1] - 1))
        return x_top, x_bottom

    # --- persistence ----------------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "H": self.H.tolist(),
            "image_size": list(self.image_size),
            "rms_px": self.rms_px,
            "n_views": self.n_views,
        }

    @classmethod
    def from_json(cls, d: dict) -> Calibration:
        return cls(
            K=np.array(d["K"]), dist=np.array(d["dist"]), H=np.array(d["H"]),
            image_size=tuple(d["image_size"]), rms_px=d["rms_px"], n_views=d["n_views"],
        )  # fmt: skip

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls.from_json(json.loads(Path(path).read_text()))
