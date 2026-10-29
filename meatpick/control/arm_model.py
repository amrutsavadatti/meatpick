"""Arm timing model: trapezoidal moves, wrist penalty, and the pick-and-place state machine.

No physics: the arm is a set of timestamps. Positions are in the belt frame (x along travel,
y across, millimetres) and the arm is fixed in that frame, like the camera. During the grip the
arm rides with the belt, so its departure point for the bin is the pick point plus the belt travel
over the grip time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ArmState(Enum):
    IDLE = "IDLE"
    MOVING_TO_PICK = "MOVING_TO_PICK"
    GRIPPING = "GRIPPING"
    MOVING_TO_BIN = "MOVING_TO_BIN"
    RELEASING = "RELEASING"


@dataclass
class ArmParams:
    vmax_mm_s: float
    amax_mm_s2: float
    grip_s: float
    release_s: float
    wrist_penalty_s: float
    wrist_threshold_deg: float
    wrist_symmetric: bool
    base_mm: tuple[float, float]
    bin_mm: tuple[float, float]
    reach_radius_mm: float
    zone_x_mm: tuple[float, float]

    @classmethod
    def from_cfg(cls, cfg: dict) -> ArmParams:
        r = cfg["robot"]
        return cls(
            vmax_mm_s=r["vmax_mps"] * 1000.0,
            amax_mm_s2=r["amax_mps2"] * 1000.0,
            grip_s=r["grip_s"],
            release_s=r["release_s"],
            wrist_penalty_s=r["wrist_rotation_penalty_s"],
            wrist_threshold_deg=r["wrist_rotation_threshold_deg"],
            wrist_symmetric=r["wrist_symmetric"],
            base_mm=tuple(r["base_mm"]),
            bin_mm=tuple(r["bin_mm"]),
            reach_radius_mm=r["reach_radius_mm"],
            zone_x_mm=tuple(r["reach_zone_x_mm"]),
        )


def move_time_s(distance_mm: float, vmax_mm_s: float, amax_mm_s2: float) -> float:
    """Accelerate, cruise, decelerate. Too short to reach vmax: accelerate then brake."""
    d = abs(distance_mm)
    if d <= 0.0:
        return 0.0
    d_ramps = vmax_mm_s**2 / amax_mm_s2  # distance covered speeding up and slowing down
    if d >= d_ramps:
        return d / vmax_mm_s + vmax_mm_s / amax_mm_s2
    return 2.0 * float(np.sqrt(d / amax_mm_s2))


def wrist_rotation_deg(from_yaw_deg: float, to_yaw_deg: float, symmetric: bool = False) -> float:
    """Wrist turn needed between two gripper yaws (both in [0, 180)). A symmetric gripper may
    take the short way round; otherwise the turn is the plain difference."""
    d = abs(to_yaw_deg - from_yaw_deg)
    return min(d, 180.0 - d) if symmetric else d


def pick_move_time_s(distance_mm: float, rotation_deg: float, p: ArmParams) -> float:
    """Translation time plus a fixed penalty when the wrist must turn past the threshold."""
    t = move_time_s(distance_mm, p.vmax_mm_s, p.amax_mm_s2)
    return t + (p.wrist_penalty_s if rotation_deg > p.wrist_threshold_deg else 0.0)


def belt_to_robot(p: ArmParams, x_mm: float, y_mm: float) -> tuple[float, float]:
    return x_mm - p.base_mm[0], y_mm - p.base_mm[1]


def robot_to_belt(p: ArmParams, x_mm: float, y_mm: float) -> tuple[float, float]:
    return x_mm + p.base_mm[0], y_mm + p.base_mm[1]


@dataclass
class SecondPick:
    """A second piece gripped on the same trip: a short move to it, then another grip."""

    track_id: int
    extra_s: float  # move_time(gap between the pieces) + grip_s


@dataclass
class ArmEvent:
    t_s: float
    kind: str  # "grip" (fingers close on the piece) | "release" | "idle"
    track_id: int


@dataclass
class _Phase:
    state: ArmState
    t_start: float
    t_end: float


@dataclass
class Arm:
    """Timestamped state machine. ``begin`` schedules a whole cycle; ``advance`` replays it."""

    p: ArmParams
    state: ArmState = ArmState.IDLE
    rest_mm: tuple[float, float] = (0.0, 0.0)  # where the arm is (or will be) when free
    yaw_deg: float = 0.0  # gripper yaw it will have when free
    free_at_s: float = 0.0
    track_id: int | None = None
    history: list[tuple[float, ArmState]] = field(default_factory=list)
    _phases: list[_Phase] = field(default_factory=list)
    _events: list[ArmEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rest_mm == (0.0, 0.0):
            self.rest_mm = self.p.bin_mm  # starts parked over the bin
        self.history.append((0.0, ArmState.IDLE))

    def begin(self, plan, belt_speed_mm_s: float, second: SecondPick | None = None) -> None:
        """Schedule a pick-and-place cycle from a feasible InterceptPlan (optionally two pieces)."""
        if self.state != ArmState.IDLE or plan.depart_t_s < self.free_at_s - 1e-9:
            raise RuntimeError(f"arm busy until {self.free_at_s:.3f} s (state {self.state.name})")
        p = self.p
        t_grip = plan.grip_at_t_s
        t_grip_done = t_grip + p.grip_s + (second.extra_s if second else 0.0)
        # the arm rides the belt while gripping, so it leaves for the bin from further downstream
        leave_xy = (plan.pick_x_mm + belt_speed_mm_s * p.grip_s, plan.pick_y_mm)
        d_bin = float(np.hypot(p.bin_mm[0] - leave_xy[0], p.bin_mm[1] - leave_xy[1]))
        t_bin = move_time_s(d_bin, p.vmax_mm_s, p.amax_mm_s2)
        t_rel = t_grip_done + t_bin
        t_end = t_rel + p.release_s
        self._phases = [
            _Phase(ArmState.MOVING_TO_PICK, plan.depart_t_s, t_grip),
            _Phase(ArmState.GRIPPING, t_grip, t_grip_done),
            _Phase(ArmState.MOVING_TO_BIN, t_grip_done, t_rel),
            _Phase(ArmState.RELEASING, t_rel, t_end),
        ]
        self._events = [ArmEvent(t_grip, "grip", plan.track_id)]
        if second:  # the second grip completes the trip's gripping phase
            self._events.append(ArmEvent(t_grip_done - p.grip_s + 1e-9, "grip", second.track_id))
        self._events += [ArmEvent(t_rel, "release", plan.track_id)]
        if second:
            self._events.append(ArmEvent(t_rel, "release", second.track_id))
        self._events.append(ArmEvent(t_end, "idle", plan.track_id))
        self.track_id, self.free_at_s = plan.track_id, t_end
        self.rest_mm, self.yaw_deg = p.bin_mm, plan.gripper_yaw_deg
        self.state = ArmState.MOVING_TO_PICK
        self.history.append((plan.depart_t_s, self.state))

    def advance(self, now_s: float) -> list[ArmEvent]:
        """Move the state machine up to ``now_s``; return the events that fired, in order."""
        fired = []
        while self._events and self._events[0].t_s <= now_s + 1e-12:
            fired.append(self._events.pop(0))
        while self._phases and self._phases[0].t_end <= now_s + 1e-12:
            done = self._phases.pop(0)
            nxt = self._phases[0].state if self._phases else ArmState.IDLE
            self.history.append((done.t_end, nxt))
        if self._phases:
            self.state = self._phases[0].state
        else:
            self.state, self.track_id = ArmState.IDLE, None
        return fired

    @property
    def busy(self) -> bool:
        return self.state != ArmState.IDLE
