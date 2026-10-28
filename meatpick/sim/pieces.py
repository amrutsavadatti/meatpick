"""Piece generator: one cut pork portion as an RGBA sprite plus exact ground truth.

Sprite frame: image x to the right, y down. The long axis is the vector
``(cos a, sin a)`` and its normal is ``(-sin a, cos a)``. A freshly generated piece has
``angle_deg == 0``; ``rotate`` turns it. ``fat_side`` is +1 when the fat strip lies on the +normal
side of the long axis, -1 on the other side. ``rotate`` keeps ``fat_side`` consistent with the
reduced angle (a half turn flips its sign).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from meatpick.sim.textures import TextureBank

ALPHA_THRESHOLD = 128  # mask = alpha >= this, so mask and sprite alpha agree exactly


@dataclass
class Piece:
    sprite: np.ndarray  # HxWx4 uint8, BGRA, straight (non-premultiplied) alpha
    mask: np.ndarray  # HxW bool, == (sprite alpha >= ALPHA_THRESHOLD)
    fat_mask: np.ndarray  # HxW bool, the fat strip inside the mask
    cx_px: float  # true center, sprite pixel coordinates
    cy_px: float
    angle_deg: float  # long axis direction in [0, 180)
    fat_side: int  # +1 or -1 relative to the long-axis normal
    length_mm: float
    width_mm: float
    thickness_mm: float
    weight_g: float
    px_per_mm: float


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _sample_dims(cfg: dict, rng: np.random.Generator) -> tuple[float, float, float, float]:
    p = cfg["product"]
    length = rng.uniform(*p["length_mm"])
    width = min(rng.uniform(*p["width_mm"]), length * 0.75)  # a clear long axis must exist
    area = length * width * 0.8
    thickness = rng.uniform(*p["weight_g"]) / (area * p["density_g_per_mm3"])
    thickness = float(np.clip(thickness, *p["thickness_mm"]))
    weight = area * thickness * p["density_g_per_mm3"]
    return length, width, thickness, weight


def _outline(
    shape: tuple[int, int], a_px: float, b_px: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft alpha in [0,1], plus the centered coordinate grids (x_px, y_px)."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x -= (w - 1) / 2
    y -= (h - 1) / 2
    n = rng.choice([2.0, rng.uniform(3.0, 5.0)])  # ellipse or rounded rectangle
    d = (np.abs(x / a_px) ** n + np.abs(y / b_px) ** n) ** (1.0 / n)
    theta = np.arctan2(y / b_px, x / a_px)
    wobble = np.zeros_like(d)
    for k in (3, 4, 5, 6):
        wobble += rng.uniform(0, 0.12 / k) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    signed_px = (d - (1.0 + wobble)) * min(a_px, b_px)  # approx distance to the edge, px
    alpha = np.clip(0.5 - signed_px, 0.0, 1.0)
    return alpha, x, y


def generate_piece(
    cfg: dict, bank: TextureBank, rng: np.random.Generator, px_per_mm: float
) -> Piece:
    length, width, thickness, weight = _sample_dims(cfg, rng)
    ppm = px_per_mm
    a_px, b_px = length * ppm / 2, width * ppm / 2
    pad = 6
    w, h = int(np.ceil(length * ppm * 1.1)) + 2 * pad, int(np.ceil(width * ppm * 1.1)) + 2 * pad
    alpha, x, y = _outline((h, w), a_px, b_px, rng)

    # Fat strip hugging one long edge: within fat_w of the outline, fading toward the ends.
    side = int(rng.choice([-1, 1]))
    fat_w_px = rng.uniform(*cfg["product"]["fat_strip_width_mm"]) * ppm
    dist = cv2.distanceTransform((alpha >= 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    wavy_w = fat_w_px + 1.5 * ppm * np.sin(x / a_px * rng.uniform(1.5, 3.5) + rng.uniform(0, 6))
    near_edge = _smoothstep((wavy_w - dist) / (2.0 * ppm) + 0.5)
    on_side = _smoothstep(side * y / (0.3 * b_px))
    along = 1.0 - _smoothstep((np.abs(x) / a_px - 0.5) / 0.35)
    fat_amount = near_edge * on_side * along

    lean = bank.sample("lean", (h, w), rng)
    fat = bank.sample("fat", (h, w), rng)
    lean = _shift_color(lean, cfg, rng)
    rgb = (
        lean.astype(np.float32) * (1 - fat_amount[..., None])
        + fat.astype(np.float32) * fat_amount[..., None]
    )
    rgb *= _thickness_shading(alpha >= 0.5, thickness * ppm, x, y, rng)[..., None]

    sprite = np.dstack(
        [np.clip(rgb, 0, 255).astype(np.uint8), (alpha * 255 + 0.5).astype(np.uint8)]
    )
    mask = sprite[..., 3] >= ALPHA_THRESHOLD
    return Piece(
        sprite=sprite,
        mask=mask,
        fat_mask=mask & (fat_amount >= 0.5),
        cx_px=(w - 1) / 2,
        cy_px=(h - 1) / 2,
        angle_deg=0.0,
        fat_side=side,
        length_mm=float(length),
        width_mm=float(width),
        thickness_mm=thickness,
        weight_g=float(weight),
        px_per_mm=ppm,
    )


def _shift_color(bgr: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Per-piece tint: hue toward pork pink, then randomly pinker or paler."""
    p = cfg["product"]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + p["lean_hue_shift"] + rng.uniform(-4, 4)) % 180
    t = rng.uniform(*p["paleness_range"])  # >0 paler, <0 pinker
    hsv[..., 1] *= 1 - 0.6 * max(t, 0) + 0.4 * max(-t, 0)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + 0.25 * t), 0, 255)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _thickness_shading(
    inside: np.ndarray, thickness_px: float, x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Edges roll off darker (rounded thick cut) plus a gentle directional light gradient."""
    dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
    rim = 1.0 - 0.14 * (1.0 - _smoothstep(dist / max(thickness_px * 0.4, 1.0)))
    ang = rng.uniform(0, 2 * np.pi)
    grad = (x * np.cos(ang) + y * np.sin(ang)) / (np.abs(x).max() + 1e-6)
    return (rim * (1.0 + 0.06 * grad)).astype(np.float32)


def rotate(piece: Piece, angle_deg: float) -> Piece:
    """Rotate the piece by ``angle_deg`` (same sense as ``Piece.angle_deg``) about its center.

    The returned canvas grows to fit; mask and fat mask are re-derived from the rotated alpha.
    """
    h, w = piece.sprite.shape[:2]
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    nw = int(np.ceil(abs(w * c) + abs(h * s))) + 2
    nh = int(np.ceil(abs(w * s) + abs(h * c))) + 2
    ncx, ncy = (nw - 1) / 2, (nh - 1) / 2
    r = np.array([[c, -s], [s, c]], np.float64)
    t = np.array([ncx, ncy]) - r @ np.array([piece.cx_px, piece.cy_px])
    m = np.hstack([r, t[:, None]])
    # Interpolate color weighted by alpha so transparent pixels do not bleed into the edge.
    a = piece.sprite[..., 3:4].astype(np.float32) / 255
    pre = np.dstack([piece.sprite[..., :3].astype(np.float32) * a, a[..., 0]])
    warped = cv2.warpAffine(pre, m, (nw, nh), flags=cv2.INTER_LINEAR)
    a2 = warped[..., 3:4]
    bgr = np.where(a2 > 1e-4, warped[..., :3] / np.maximum(a2, 1e-4), 0)
    sprite = np.dstack([np.clip(bgr, 0, 255), a2 * 255]).round().astype(np.uint8)
    fat = cv2.warpAffine(piece.fat_mask.astype(np.float32), m, (nw, nh), flags=cv2.INTER_LINEAR)
    mask = sprite[..., 3] >= ALPHA_THRESHOLD
    total = piece.angle_deg + angle_deg
    flips = int(np.floor(total / 180.0))  # each half turn swaps which side the normal points to
    return Piece(
        sprite=sprite,
        mask=mask,
        fat_mask=mask & (fat >= 0.5),
        cx_px=ncx,
        cy_px=ncy,
        angle_deg=float(total % 180.0),
        fat_side=piece.fat_side * (-1) ** flips,
        length_mm=piece.length_mm,
        width_mm=piece.width_mm,
        thickness_mm=piece.thickness_mm,
        weight_g=piece.weight_g,
        px_per_mm=piece.px_per_mm,
    )


def piece_grid(pieces: list[Piece], cols: int = 10, cell: int = 300) -> np.ndarray:
    """Pieces on a belt-blue background, one per cell, for eyeballing."""
    bg = np.array([140, 90, 30], np.uint8)
    rows = (len(pieces) + cols - 1) // cols
    canvas = np.tile(bg, (rows * cell, cols * cell, 1))
    for i, p in enumerate(pieces):
        spr = p.sprite
        scale = min(1.0, (cell - 8) / max(spr.shape[:2]))
        if scale < 1.0:
            spr = cv2.resize(spr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = spr.shape[:2]
        y0, x0 = (i // cols) * cell + (cell - h) // 2, (i % cols) * cell + (cell - w) // 2
        a = spr[..., 3:4].astype(np.float32) / 255
        roi = canvas[y0 : y0 + h, x0 : x0 + w].astype(np.float32)
        canvas[y0 : y0 + h, x0 : x0 + w] = (spr[..., :3] * a + roi * (1 - a)).astype(np.uint8)
    return canvas
