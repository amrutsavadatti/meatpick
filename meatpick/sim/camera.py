"""Camera renderer: belt texture, composited pieces, lens and sensor effects, plus frame truth.

The scene is composited in the ideal (undistorted) image frame on a canvas that is larger than the
frame, then warped once to the final image with a tilt homography and OpenCV radial distortion.
The same forward model (``Camera.project``) maps ground-truth points to final pixels, so truth
polygons and rendered pixels agree. Lighting, white balance and sensor noise act on the final
image. Frames are triggered by encoder distance, not time.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from meatpick.sim import geometry
from meatpick.sim.textures import _value_noise
from meatpick.sim.world import World, touching_flags
from meatpick.types import Frame

TILE_MM = 1000  # belt texture repeats (belt splice) every metre of travel


@dataclass
class PieceFrameTruth:
    id: int
    polygon_px: np.ndarray  # Nx2 outline of the whole piece in final image pixels
    center_px: np.ndarray  # (u, v) of the true center
    x_mm: float  # belt-frame center at capture
    y_mm: float
    angle_deg: float  # long axis, belt frame, [0, 180)
    fat_side: int
    visibility: float  # visible fraction of the piece in this frame, 0-1
    touching: bool
    length_mm: float
    width_mm: float
    weight_g: float


@dataclass
class FrameTruth:
    encoder: int
    t_s: float
    pieces: list[PieceFrameTruth]
    instance_map: np.ndarray  # HxW int32, piece id per pixel, -1 for belt
    glare_level: float  # fraction of visible meat pixels that are blown out


def _seamless_rows(tile: np.ndarray) -> np.ndarray:
    """Cross-fade a texture with a half-shifted copy so it wraps without a visible join."""
    h = tile.shape[0]
    w = 1.0 - np.abs(2.0 * np.arange(h) / h - 1.0)
    w = w.reshape((h,) + (1,) * (tile.ndim - 1)).astype(np.float32)
    return tile * w + np.roll(tile, h // 2, axis=0) * (1 - w)


class Camera:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        c = cfg["camera"]
        self.cfg, self.rng = cfg, rng
        self.W, self.H = c["width_px"], c["height_px"]
        self.ppm = geometry.px_per_mm(cfg)
        self.margin = c["canvas_margin_px"]
        self.fx = c["effects"]
        self.trigger_counts = int(
            round(c["trigger_every_mm"] / cfg["belt"]["mm_per_encoder_count"])
        )
        self._next_trigger = self.trigger_counts
        self._draw_parameters()
        self._build_belt_texture()
        self._build_lens_maps()
        self._build_lighting()
        self._exposure_state = 0.0
        self._wb_state = np.zeros(3, np.float32)

    # --- setup ----------------------------------------------------------------------------
    def _jitter(self, value: float, rel: float = 0.3) -> float:
        return (
            value * float(self.rng.uniform(1 - rel, 1 + rel))
            if self.cfg["camera"]["randomize"]
            else value
        )

    def _draw_parameters(self) -> None:
        c = self.cfg["camera"]
        r = self.cfg["camera"]["randomize"]
        self.f_px = self._jitter(c["f_px"], 0.05)
        self.tilt = np.array(c["tilt_deg"], float) * (self.rng.uniform(0.5, 1.5, 2) if r else 1)
        self.dist = np.array(c["dist_k"], float) * (self.rng.uniform(0.6, 1.4, 2) if r else 1)
        if not self.fx["perspective"]:
            self.tilt = np.zeros(2)
        if not self.fx["lens_distortion"]:
            self.dist = np.zeros(2)
        self.vignette = self._jitter(c["vignette"]) if self.fx["vignetting"] else 0.0
        self.gradient = self._jitter(c["light_gradient"]) if self.fx["vignetting"] else 0.0
        self.noise_sigma = self._jitter(c["noise_sigma"], 0.5) if self.fx["sensor_noise"] else 0.0
        self.glare_strength = self._jitter(c["glare_strength"]) if self.fx["glare"] else 0.0

    def _intrinsics(self) -> np.ndarray:
        return np.array(
            [[self.f_px, 0, self.W / 2], [0, self.f_px, self.H / 2], [0, 0, 1]], np.float64
        )

    def _build_lens_maps(self) -> None:
        """Tilt homography then radial distortion; the remap maps are its inverse."""
        K = self._intrinsics()
        pitch, yaw = np.radians(self.tilt)
        rx = np.array(
            [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
        )
        ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        self.K = K
        self.H_tilt = K @ ry @ rx @ np.linalg.inv(K)  # ideal frame px -> pre-distortion px
        self.dist_coeffs = np.array([self.dist[0], self.dist[1], 0.0, 0.0, 0.0])
        gy, gx = np.mgrid[0 : self.H, 0 : self.W].astype(np.float32)
        out_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
        pre = cv2.undistortPoints(out_pts, K, self.dist_coeffs, P=K).reshape(-1, 2)
        ideal = cv2.perspectiveTransform(
            pre.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(self.H_tilt)
        )
        ideal = ideal.reshape(self.H, self.W, 2).astype(np.float32) + self.margin
        self.map_x, self.map_y = ideal[..., 0].copy(), ideal[..., 1].copy()

    def project(self, ideal_px: np.ndarray) -> np.ndarray:
        """Ideal frame pixels (u, v), Nx2 -> final image pixels. Same model the maps invert."""
        pts = np.asarray(ideal_px, np.float64).reshape(-1, 1, 2)
        tilted = cv2.perspectiveTransform(pts, self.H_tilt).reshape(-1, 2)
        norm = (tilted - [self.W / 2, self.H / 2]) / self.f_px
        r2 = (norm**2).sum(1, keepdims=True)
        k1, k2 = self.dist
        return norm * (1 + k1 * r2 + k2 * r2**2) * self.f_px + [self.W / 2, self.H / 2]

    def belt_to_px(self, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        """Belt-frame mm at the camera reference -> final image pixels."""
        ppm, bw = self.ppm, self.cfg["belt"]["width_mm"]
        ideal = np.stack([(bw - np.asarray(y_mm)) * ppm, np.asarray(x_mm) * ppm], axis=1)
        return self.project(ideal)

    def _build_belt_texture(self) -> None:
        rng, cfg = self.rng, self.cfg
        wc = self.W + 2 * self.margin
        ht = int(round(TILE_MM * self.ppm))
        shape = (ht, wc)
        base = np.array(cfg["belt"]["color_bgr"], np.float32)
        n = _value_noise(shape, rng, octaves=5, base_cells=4)
        tex = base[None, None, :] * (0.88 + 0.2 * n[..., None])
        # weave: faint fibres running along travel (image rows) and a fine cross hatch
        fibre = _value_noise(shape, rng, octaves=3, base_cells=64, anisotropy=0.02)
        yy, xx = np.mgrid[0:ht, 0:wc].astype(np.float32)
        hatch = 0.5 * (np.sin(xx * 2 * np.pi / 7) + np.sin(yy * 2 * np.pi / 7))
        tex *= (1 + 0.05 * (fibre - 0.5) + 0.012 * hatch)[..., None]
        # wear: pale scuffs streaked along travel
        scuff = _value_noise(shape, rng, octaves=4, base_cells=16, anisotropy=0.05)
        wear = np.clip((scuff - 0.7) / 0.15, 0, 1)[..., None]
        tex = tex * (1 - 0.25 * wear) + np.array([185, 150, 110], np.float32) * 0.25 * wear
        # juice stains: dark red-brown blotches
        stain = _value_noise(shape, rng, octaves=4, base_cells=14)
        s = np.clip((stain - 0.76) / 0.1, 0, 1)
        s = cv2.GaussianBlur(s, (0, 0), 3)[..., None]
        tex = tex * (1 - 0.3 * s) + np.array([30, 50, 120], np.float32) * 0.3 * s
        tex = _seamless_rows(tex.astype(np.float32))
        # belt splice: a thin dark gap with a pale edge, crossing the belt once per tile
        tex[0:6] *= 0.5
        tex[6:9] = tex[6:9] * 0.8 + 45
        self.belt_tex = tex
        self.tile_rows = ht

    def _build_lighting(self) -> None:
        yy, xx = np.mgrid[0 : self.H, 0 : self.W].astype(np.float32)
        nx, ny = (xx - self.W / 2) / (self.W / 2), (yy - self.H / 2) / (self.H / 2)
        field = 1.0 - self.vignette * 0.5 * (nx**2 + ny**2)
        ang = self.rng.uniform(0, 2 * np.pi)
        field *= 1.0 + self.gradient * (nx * np.cos(ang) + ny * np.sin(ang))
        self.light_field = field.astype(np.float32)
        # glare lobes: fixed lamp reflections in the ideal frame (canvas coordinates)
        hc, wc = self.H + 2 * self.margin, self.W + 2 * self.margin
        gy, gx = np.mgrid[0:hc, 0:wc].astype(np.float32)
        lobes = np.zeros((hc, wc), np.float32)
        for _ in range(self.cfg["camera"]["glare_lobes"]):
            cx, cy = self.rng.uniform(0.15, 0.85) * wc, self.rng.uniform(0.15, 0.85) * hc
            sig = self.rng.uniform(90, 200)
            lobes += self.rng.uniform(0.6, 1.0) * np.exp(
                -((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sig**2)
            )
        self.glare_lobes = np.clip(lobes, 0, 1)

    # --- rendering ------------------------------------------------------------------------
    def _background(self, travel_mm: float) -> np.ndarray:
        shift = int(round(travel_mm * self.ppm))
        hc = self.H + 2 * self.margin
        rows = (np.arange(hc) - self.margin - shift) % self.tile_rows
        return self.belt_tex[rows].copy()

    def _composite(self, canvas: np.ndarray, world: World) -> tuple[np.ndarray, dict[int, int]]:
        hc, wc = canvas.shape[:2]
        inst = np.full((hc, wc), -1, np.float32)
        full_px: dict[int, int] = {}
        self._rendered_center: dict[int, tuple[float, float]] = {}
        pad = 14
        for p in sorted(world.pieces, key=lambda q: q.id):
            spr = p.sprite
            u, v = geometry.belt_to_image(self.cfg, p.x_mm, p.y_mm)
            x0 = int(round(u + self.margin - spr.cx_px))
            y0 = int(round(v + self.margin - spr.cy_px))
            h, w = spr.sprite.shape[:2]
            # sprites are drawn at whole pixels; truth must describe what was drawn
            self._rendered_center[p.id] = (
                x0 + spr.cx_px - self.margin,
                y0 + spr.cy_px - self.margin,
            )
            ry0, ry1 = max(y0 - pad, 0), min(y0 + h + pad, hc)
            rx0, rx1 = max(x0 - pad, 0), min(x0 + w + pad, wc)
            if ry0 >= ry1 or rx0 >= rx1:
                continue
            alpha = np.zeros((ry1 - ry0, rx1 - rx0), np.float32)
            rgb = np.zeros(alpha.shape + (3,), np.float32)
            gloss = np.zeros_like(alpha)
            sy0, sy1 = max(y0, ry0), min(y0 + h, ry1)
            sx0, sx1 = max(x0, rx0), min(x0 + w, rx1)
            if sy0 >= sy1 or sx0 >= sx1:
                continue
            ys, xs = slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0)
            yd, xd = slice(sy0 - ry0, sy1 - ry0), slice(sx0 - rx0, sx1 - rx0)
            alpha[yd, xd] = spr.sprite[ys, xs, 3] / 255.0
            rgb[yd, xd] = spr.sprite[ys, xs, :3]
            gloss[yd, xd] = spr.gloss[ys, xs]
            # soft contact shadow, light from the upper left
            shifted = np.zeros_like(alpha)  # zero-filled shift: np.roll would wrap at the ROI edge
            shifted[6:, 4:] = alpha[:-6, :-4]
            shadow = cv2.GaussianBlur(shifted, (0, 0), 5) * 0.4
            roi = canvas[ry0:ry1, rx0:rx1]
            roi *= (1 - shadow)[..., None]
            if self.glare_strength > 0:
                lobe = self.glare_lobes[ry0:ry1, rx0:rx1]
                rgb += (self.glare_strength * 255 * lobe * gloss)[..., None] * np.array(
                    [0.95, 1.0, 1.0], np.float32
                )
            roi[:] = roi * (1 - alpha[..., None]) + np.clip(rgb, 0, 255) * alpha[..., None]
            m = np.zeros(alpha.shape, bool)
            m[yd, xd] = spr.mask[ys, xs]
            inst[ry0:ry1, rx0:rx1][m] = p.id
            full_px[p.id] = int(spr.mask.sum())
        return inst, full_px

    def _motion_blur(self, canvas: np.ndarray, speed_mps: float) -> np.ndarray:
        length = speed_mps * 1000.0 * self.cfg["camera"]["exposure_s"] * self.ppm
        if not self.fx["motion_blur"] or length < 0.3:
            return canvas
        n = int(np.ceil(length)) | 1  # odd tap count centred on the pixel
        k = np.zeros(n, np.float32)
        k[:] = 1.0
        extra = n - length
        k[0] -= extra / 2
        k[-1] -= extra / 2
        k = np.clip(k, 0, None)
        k /= k.sum()
        return cv2.filter2D(canvas, -1, k.reshape(-1, 1), borderType=cv2.BORDER_REPLICATE)

    def render(self, world: World) -> tuple[Frame, FrameTruth]:
        canvas = self._background(world.belt.travel_mm)
        inst_canvas, full_px = self._composite(canvas, world)
        canvas = self._motion_blur(canvas, world.belt.speed_mps)
        img = cv2.remap(
            canvas, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        inst = cv2.remap(inst_canvas, self.map_x, self.map_y, cv2.INTER_NEAREST, borderValue=-1)
        inst = inst.astype(np.int32)

        c = self.cfg["camera"]
        img *= self.light_field[..., None]
        if self.fx["exposure_drift"]:
            self._exposure_state = 0.97 * self._exposure_state + self.rng.normal(
                0, c["exposure_drift"] * 0.25
            )
            self._wb_state = 0.97 * self._wb_state + self.rng.normal(
                0, c["wb_drift"] * 0.25, 3
            ).astype(np.float32)
            img *= 1.0 + np.clip(self._exposure_state, -c["exposure_drift"], c["exposure_drift"])
            img *= 1.0 + np.clip(self._wb_state, -c["wb_drift"], c["wb_drift"])
        if self.noise_sigma > 0:
            img += self.rng.standard_normal(img.shape, dtype=np.float32) * self.noise_sigma
        frame_img = np.clip(img, 0, 255).astype(np.uint8)

        truth = self._frame_truth(world, inst, full_px, frame_img)
        return Frame(frame_img, world.belt.encoder, world.t_s), truth

    def _frame_truth(
        self, world: World, inst: np.ndarray, full_px: dict[int, int], img: np.ndarray
    ) -> FrameTruth:
        gts = {g.id: g for g in world.ground_truth()}
        touching = touching_flags(list(gts.values()), self.cfg["robot"]["touching_clearance_mm"])
        counts = dict(zip(*np.unique(inst[inst >= 0], return_counts=True), strict=True))
        out = []
        for p in world.pieces:
            n_vis = int(counts.get(p.id, 0))
            if n_vis < self.cfg["camera"]["min_visible_px"]:
                continue
            g = gts[p.id]
            rc = np.array(self._rendered_center[p.id])
            rel_px = np.stack([-p.rel_polygon_mm[:, 1], p.rel_polygon_mm[:, 0]], axis=1) * self.ppm
            poly = self.project(rc + rel_px)
            ideal_area = cv2.contourArea((p.rel_polygon_mm * self.ppm).astype(np.float32))
            area_scale = cv2.contourArea(poly.astype(np.float32)) / max(ideal_area, 1e-6)
            vis = min(1.0, n_vis / max(full_px.get(p.id, 1) * area_scale, 1.0))
            center = self.project(rc[None, :])[0]
            out.append(
                PieceFrameTruth(
                    id=p.id,
                    polygon_px=poly,
                    center_px=center,
                    x_mm=g.x_mm,
                    y_mm=g.y_mm,
                    angle_deg=g.angle_deg,
                    fat_side=g.fat_side,
                    visibility=float(vis),
                    touching=touching[p.id],
                    length_mm=g.length_mm,
                    width_mm=g.width_mm,
                    weight_g=g.weight_g,
                )
            )
        meat = inst >= 0
        glare = float((img[meat].max(axis=1) >= 250).mean()) if meat.any() else 0.0
        return FrameTruth(world.belt.encoder, world.t_s, out, inst, glare)

    # --- triggering -----------------------------------------------------------------------
    def maybe_capture(self, world: World) -> tuple[Frame, FrameTruth] | None:
        """Capture when the encoder has passed the next trigger point (every N counts of travel)."""
        enc = world.belt.encoder
        if enc < self._next_trigger:
            return None
        self._next_trigger = (enc // self.trigger_counts + 1) * self.trigger_counts
        return self.render(world)
