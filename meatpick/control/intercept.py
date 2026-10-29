"""Intercept planner: where and when can the arm meet a piece that is moving with the belt?

The grip time ``t`` and the piece position at ``t`` depend on each other: the later the grip, the
further downstream the piece, the longer the move. We iterate ``t -> piece position -> move time``
until it settles (the PRD allows 3 iterations and 1 ms). The arm also waits if the piece has not
yet entered the reach zone, and the plan is rejected if the piece will already be past it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from meatpick.control.arm_model import (
    ArmParams,
    belt_to_robot,
    pick_move_time_s,
    wrist_rotation_deg,
)
from meatpick.types import PickCommand, Track


@dataclass
class InterceptPlan:
    track_id: int
    feasible: bool
    reason: str = ""  # why not feasible: outside_reach | too_late | belt_stopped
    pick_x_mm: float = 0.0  # belt frame, where the piece will be at grip time
    pick_y_mm: float = 0.0
    gripper_yaw_deg: float = 0.0
    grip_at_t_s: float = 0.0
    grip_at_encoder: int = 0
    depart_t_s: float = 0.0  # when the arm leaves for the piece
    move_s: float = 0.0
    wrist_rotation_deg: float = 0.0
    iterations: int = 0
    converged: bool = True

    def to_command(self, p: ArmParams, place: str = "BIN") -> PickCommand:
        rx, ry = belt_to_robot(p, self.pick_x_mm, self.pick_y_mm)
        return PickCommand(
            track_id=self.track_id,
            pick_x_mm=rx,
            pick_y_mm=ry,
            gripper_yaw_deg=self.gripper_yaw_deg,
            grip_at_encoder=self.grip_at_encoder,
            grip_at_t_s=self.grip_at_t_s,
            place=place,
        )


def _time_until_travel(travel: Callable[[float], float], distance_mm: float) -> float:
    """Smallest dt with travel(dt) >= distance (travel is non-decreasing); inf if never."""
    if distance_mm <= 0.0:
        return 0.0
    hi = 60.0
    if travel(hi) < distance_mm:
        return float("inf")
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if travel(mid) < distance_mm else (lo, mid)
    return hi


def plan_intercept(
    track: Track,
    now_s: float,
    encoder_now: int,
    arm_free_at_s: float,
    arm_rest_mm: tuple[float, float],
    arm_yaw_deg: float,
    p: ArmParams,
    cfg: dict,
    belt_speed_mm_s: float,
    travel_fn: Callable[[float], float] | None = None,
) -> InterceptPlan:
    """``travel_fn(dt)`` predicts belt travel over the next ``dt`` seconds; the default assumes the
    belt keeps its current speed. A belt controller that is about to change speed should pass a
    better predictor, or the arm will meet the piece where it would have been."""
    travel = travel_fn or (lambda dt: belt_speed_mm_s * dt)
    mm_per_count = cfg["belt"]["mm_per_encoder_count"]
    max_iter, tol = cfg["control"]["intercept_max_iter"], cfg["control"]["intercept_tol_s"]
    plan = InterceptPlan(track.id, feasible=False)

    x_now = track.x_mm + (encoder_now - track.ref_encoder) * mm_per_count
    y = track.y_mm
    zone_lo, zone_hi = p.zone_x_mm
    yaw = (track.angle_deg + cfg["robot"]["gripper_yaw_offset_deg"]) % 180.0
    rot = wrist_rotation_deg(arm_yaw_deg, yaw, p.wrist_symmetric)
    t0 = max(now_s, arm_free_at_s)  # the arm cannot leave before it is free
    ax, ay = arm_rest_mm
    plan.gripper_yaw_deg, plan.wrist_rotation_deg, plan.depart_t_s = yaw, rot, t0

    if x_now > zone_hi:
        plan.reason = "too_late"
        return plan
    dt_enter = _time_until_travel(travel, zone_lo - x_now)
    if not np.isfinite(dt_enter):
        plan.reason = "belt_stopped"  # the piece will never reach the zone
        return plan
    t_enter = now_s + dt_enter  # no grip before the piece is inside the reach zone

    def grip_time_for(t_grip: float) -> float:
        xg = x_now + travel(t_grip - now_s)
        move = pick_move_time_s(float(np.hypot(xg - ax, y - ay)), rot, p)
        return max(t0 + move, t_enter)

    t_grip = grip_time_for(t0)
    converged = False
    for i in range(1, max_iter + 1):
        nxt = grip_time_for(t_grip)
        converged = abs(nxt - t_grip) < tol
        t_grip = nxt
        plan.iterations = i
        if converged:
            break
    plan.converged = converged

    x_grip = x_now + travel(t_grip - now_s)
    plan.pick_x_mm, plan.pick_y_mm, plan.grip_at_t_s = x_grip, y, t_grip
    plan.move_s = max(t_grip - t0, 0.0)
    plan.grip_at_encoder = encoder_now + int(round(travel(t_grip - now_s) / mm_per_count))
    if x_grip > zone_hi:
        plan.reason = "too_late"
    elif np.hypot(x_grip - p.base_mm[0], y - p.base_mm[1]) > p.reach_radius_mm:
        plan.reason = "outside_reach"
    else:
        plan.feasible = True
    return plan
