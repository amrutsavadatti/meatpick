"""Belt, arrival process and world state. This is the only place ground truth lives."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from meatpick.sim import geometry
from meatpick.sim.pieces import Piece, generate_piece, rotate
from meatpick.sim.textures import TextureBank


class Belt:
    """Belt with a speed state (acceleration limited) and an encoder count."""

    def __init__(self, cfg: dict):
        b = cfg["belt"]
        self.mm_per_count = b["mm_per_encoder_count"]
        self.speed_min, self.speed_max = b["speed_min_mps"], b["speed_max_mps"]
        self.accel_max = b["accel_max_mps2"]
        self.speed_mps = b["speed_nominal_mps"]
        self.target_mps = self.speed_mps
        self.travel_mm = 0.0  # total belt travel since t = 0

    @property
    def encoder(self) -> int:
        return int(self.travel_mm / self.mm_per_count + 1e-6)

    def command(self, speed_mps: float) -> None:
        self.target_mps = float(np.clip(speed_mps, self.speed_min, self.speed_max))

    def step(self, dt_s: float) -> float:
        """Advance one step; returns the travel in mm (trapezoidal over the step)."""
        dv = self.target_mps - self.speed_mps
        new = self.speed_mps + np.clip(dv, -self.accel_max * dt_s, self.accel_max * dt_s)
        d_mm = 0.5 * (self.speed_mps + new) * dt_s * 1000.0
        self.speed_mps = float(new)
        self.travel_mm += d_mm
        return d_mm


@dataclass
class PieceTruth:
    """Ground truth for one piece at one instant, in the belt frame."""

    id: int
    x_mm: float
    y_mm: float
    angle_deg: float  # long axis in the belt frame, [0, 180)
    fat_side: int
    polygon_mm: np.ndarray  # Nx2 belt-frame outline
    state: str  # on_belt | picked | missed
    length_mm: float
    width_mm: float
    weight_g: float


@dataclass
class WorldPiece:
    id: int
    sprite: Piece  # rotated sprite, angle_deg is the image-frame angle
    x_mm: float
    y_mm: float
    theta_deg: float  # belt-frame long axis
    rel_polygon_mm: np.ndarray  # outline relative to the center, belt axes
    state: str = "on_belt"
    spawn_t_s: float = 0.0
    picked_t_s: float | None = None
    half_x_mm: float = 0.0  # half extent along travel, for placement and visibility


@dataclass
class World:
    cfg: dict
    bank: TextureBank
    rng: np.random.Generator
    belt: Belt = field(init=False)
    t_s: float = 0.0
    pieces: list[WorldPiece] = field(default_factory=list)
    n_spawned: int = 0
    n_picked: int = 0
    n_missed: int = 0

    def __post_init__(self) -> None:
        self.belt = Belt(self.cfg)
        self.ppm = geometry.px_per_mm(self.cfg)
        self._next_id = 0
        self._next_event_t = self._draw_gap()

    # --- arrivals -------------------------------------------------------------------------
    def _mean_cluster_size(self) -> float:
        a = self.cfg["arrivals"]
        if not a["bursty"]:
            return 1.0
        lo, hi = a["burst_size"]
        return (1 - a["burst_prob"]) + a["burst_prob"] * (lo + hi) / 2

    def _draw_gap(self) -> float:
        """Time to the next arrival event; events/s chosen so pieces/min hits the target."""
        rate_events_s = self.cfg["arrivals"]["rate_per_min"] / 60.0 / self._mean_cluster_size()
        return float(self.rng.exponential(1.0 / rate_events_s))

    def _make_piece(self, x_mm: float, y_mm: float) -> WorldPiece:
        base = generate_piece(self.cfg, self.bank, self.rng, self.ppm)
        theta = float(self.rng.uniform(0, 180))
        spr = rotate(base, geometry.belt_angle_to_image(theta))
        contours, _ = cv2.findContours(
            spr.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnt = max(contours, key=cv2.contourArea)
        cnt = cv2.approxPolyDP(cnt, 1.0, True)[:, 0, :].astype(float)
        du, dv = cnt[:, 0] - spr.cx_px, cnt[:, 1] - spr.cy_px
        rel = np.stack([dv / self.ppm, -du / self.ppm], axis=1)  # (dx_belt, dy_belt)
        piece = WorldPiece(
            id=self._next_id,
            sprite=spr,
            x_mm=x_mm,
            y_mm=y_mm,
            theta_deg=theta,
            rel_polygon_mm=rel,
            spawn_t_s=self.t_s,
            half_x_mm=float(np.abs(rel[:, 0]).max()),
        )
        self._next_id += 1
        self.n_spawned += 1
        return piece

    def _spawn_event(self) -> None:
        a, w, belt_w = self.cfg["arrivals"], self.cfg["world"], self.cfg["belt"]["width_mm"]
        lo_y, hi_y = w["y_margin_mm"], belt_w - w["y_margin_mm"]
        size = 1
        if a["bursty"] and self.rng.random() < a["burst_prob"]:
            size = int(self.rng.integers(a["burst_size"][0], a["burst_size"][1] + 1))
        x, y = float(w["spawn_x_mm"]), float(self.rng.uniform(lo_y, hi_y))
        prev: WorldPiece | None = None
        for _ in range(size):
            if prev is not None:  # cluster member sits just behind the previous one
                y = float(np.clip(prev.y_mm + self.rng.uniform(-40, 40), lo_y, hi_y))
            piece = self._make_piece(x, y)
            if prev is not None:
                gap = float(self.rng.uniform(*a["burst_spacing_mm"]))
                piece.x_mm = prev.x_mm - prev.half_x_mm - piece.half_x_mm - gap
            self.pieces.append(piece)
            prev = piece

    # --- stepping -------------------------------------------------------------------------
    def step(self, dt_s: float | None = None) -> float:
        """Advance the world one fixed step; returns the belt travel in mm."""
        dt = dt_s if dt_s is not None else self.cfg["world"]["dt_s"]
        travel = self.belt.step(dt)
        for p in self.pieces:
            p.x_mm += travel
        self.t_s += dt
        while self.t_s >= self._next_event_t:
            self._spawn_event()
            self._next_event_t += self._draw_gap()
        reach_end = self.cfg["robot"]["reach_zone_x_mm"][1]
        end_x = self.cfg["world"]["end_x_mm"]
        for p in self.pieces:
            if p.state == "on_belt" and p.x_mm > reach_end:
                p.state = "missed"
                self.n_missed += 1
        self.pieces = [
            p for p in self.pieces if p.state != "picked" and p.x_mm - p.half_x_mm < end_x
        ]
        return travel

    def pick(self, piece_id: int) -> bool:
        """Remove a piece at grip time. Returns False if it is not available to pick."""
        for p in self.pieces:
            if p.id == piece_id and p.state == "on_belt":
                p.state = "picked"
                p.picked_t_s = self.t_s
                self.n_picked += 1
                return True
        return False

    def ground_truth(self) -> list[PieceTruth]:
        out = []
        for p in self.pieces:
            poly = p.rel_polygon_mm + np.array([p.x_mm, p.y_mm])
            out.append(
                PieceTruth(
                    id=p.id,
                    x_mm=p.x_mm,
                    y_mm=p.y_mm,
                    angle_deg=p.theta_deg,
                    fat_side=geometry.fat_side_image_to_belt(p.sprite.angle_deg, p.sprite.fat_side),
                    polygon_mm=poly,
                    state=p.state,
                    length_mm=p.sprite.length_mm,
                    width_mm=p.sprite.width_mm,
                    weight_g=p.sprite.weight_g,
                )
            )
        return out
