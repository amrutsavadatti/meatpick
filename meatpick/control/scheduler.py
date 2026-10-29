"""Scheduler: which piece should the free arm go for next, and the command that says so.

Earliest-deadline-first over confirmed, reachable tracks. A piece is skipped when the gripper's
fingers (which straddle it across its short axis) would hit a neighbour; one that can no longer
be reached before it leaves the zone is a miss. Each pick becomes a ``PickCommand`` and a line in
the JSON-lines command log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meatpick.control.arm_model import (
    Arm,
    ArmParams,
    SecondPick,
    move_time_s,
    wrist_rotation_deg,
)
from meatpick.control.intercept import InterceptPlan, plan_intercept
from meatpick.types import PickCommand, Track

SCHEMA_VERSION = 1


# --- the command interface --------------------------------------------------------------
def command_record(
    cmd: PickCommand, issued_at_s: float, also_pick: list[int], collision_risk: bool = False
) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "issued_at_s": round(issued_at_s, 6),
        "track_id": cmd.track_id,
        "pick_x_mm": round(cmd.pick_x_mm, 3),
        "pick_y_mm": round(cmd.pick_y_mm, 3),
        "gripper_yaw_deg": round(cmd.gripper_yaw_deg, 3),
        "grip_at_encoder": cmd.grip_at_encoder,
        "grip_at_t_s": round(cmd.grip_at_t_s, 6),
        "place": cmd.place,
        "also_pick": also_pick,
        "collision_risk": collision_risk,
    }


def validate_command_record(rec: dict, cfg: dict) -> list[str]:
    """Problems with a command record; an empty list means it is valid."""
    errs = []

    def need(key, kinds, ok=lambda v: True, why=""):
        v = rec.get(key)
        is_bool_ok = kinds is bool and isinstance(v, bool)
        if (
            key not in rec
            or (isinstance(v, bool) and not is_bool_ok)
            or not isinstance(v, kinds)
            or not ok(v)
        ):
            errs.append(f"{key}={v!r} {why}".strip())

    num = (int, float)
    need("schema", int, lambda v: v == SCHEMA_VERSION, f"must be {SCHEMA_VERSION}")
    need("issued_at_s", num, lambda v: v >= 0, "must be >= 0")
    need("track_id", int, lambda v: v >= 0, "must be a non-negative integer")
    need("pick_x_mm", num)
    need("pick_y_mm", num)
    need("gripper_yaw_deg", num, lambda v: 0 <= v < 180, "must be in [0, 180)")
    need("grip_at_encoder", int, lambda v: v >= 0, "must be a non-negative integer")
    need(
        "grip_at_t_s",
        num,
        lambda v: v >= rec.get("issued_at_s", 0) - 1e-6,
        "must not precede issue",
    )
    need("place", str, lambda v: v == "BIN", "must be BIN")
    need("also_pick", list, lambda v: all(isinstance(i, int) and i >= 0 for i in v), "ids")
    need("collision_risk", bool, why="must be true or false")
    if not errs:
        r = float(np.hypot(rec["pick_x_mm"], rec["pick_y_mm"]))  # robot frame: origin at the base
        if r > cfg["robot"]["reach_radius_mm"] + 1e-6:
            errs.append(f"pick point {r:.0f} mm from the base is outside reach")
    return errs


class CommandLog:
    """Append-only JSON-lines log of issued commands."""

    def __init__(self, path: str | Path | None, cfg: dict):
        self.path, self.cfg = (Path(path) if path else None), cfg
        self.records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def write(
        self,
        cmd: PickCommand,
        issued_at_s: float,
        also_pick: list[int],
        collision_risk: bool = False,
    ) -> dict:
        rec = command_record(cmd, issued_at_s, also_pick, collision_risk)
        problems = validate_command_record(rec, self.cfg)
        if problems:  # never emit a command the robot interface would reject
            raise ValueError(f"invalid command for track {cmd.track_id}: {problems}")
        self.records.append(rec)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec


def read_commands(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# --- geometry -----------------------------------------------------------------------------
def fingers_collide(track: Track, others: list[Track], cfg: dict) -> bool:
    """Do the two fingers, standing either side of the piece's long edges, hit a neighbour?

    The fingers straddle the piece across its short axis, so a neighbour beside a long edge is in
    the way and one at the end of the long axis is not. A neighbour is approximated by a disk.
    """
    r = cfg["robot"]
    a = np.radians(track.angle_deg)
    axis, normal = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    centre = np.array([track.x_mm, track.y_mm])
    half_w = (track.width_mm or 80.0) / 2.0
    for side in (+1.0, -1.0):
        finger = centre + side * normal * (half_w + r["finger_offset_mm"])
        for o in others:
            if o.id == track.id:
                continue
            v = np.array([o.x_mm, o.y_mm]) - finger
            dx = max(abs(float(v @ axis)) - r["finger_span_mm"], 0.0)
            dy = max(abs(float(v @ normal)) - r["finger_depth_mm"], 0.0)
            radius = 0.5 * min(o.width_mm or 60.0, o.length_mm or 100.0)
            if float(np.hypot(dx, dy)) < radius:
                return True
    return False


# --- the scheduler ------------------------------------------------------------------------
@dataclass
class Dispatch:
    plan: InterceptPlan
    command: PickCommand
    record: dict
    second_plan: InterceptPlan | None = None

    @property
    def track_ids(self) -> list[int]:
        return [self.plan.track_id] + ([self.second_plan.track_id] if self.second_plan else [])


@dataclass
class Scheduler:
    cfg: dict
    log: CommandLog | None = None
    issued: set[int] = field(default_factory=set)
    skip_reasons: dict[int, str] = field(
        default_factory=dict
    )  # first reason a track was passed over
    missed: dict[int, float] = field(default_factory=dict)  # track id -> time it became a miss

    def __post_init__(self) -> None:
        self.p = ArmParams.from_cfg(self.cfg)
        self.log = self.log or CommandLog(None, self.cfg)

    def unpickable(self, tracks: list[Track]) -> set[int]:
        """Tracks the scheduler would pass over right now however long the belt waits: fingers
        blocked by a neighbour, or too unsure. The belt controller must not slow the belt for
        them (a piece nobody will pick would hold the belt forever)."""
        floor = self.cfg["control"]["min_confidence"]
        if self.cfg["control"]["last_resort_blocked"]:  # blocked pieces will be tried near the end
            return {t.id for t in tracks if t.confidence < floor}
        return {
            t.id for t in tracks
            if t.confidence < floor or (t.touching and fingers_collide(t, tracks, self.cfg))
        }  # fmt: skip

    def _skip(self, track_id: int, reason: str) -> None:
        self.skip_reasons.setdefault(track_id, reason)

    def step(
        self,
        now_s: float,
        encoder_now: int,
        tracks: list[Track],
        arm: Arm,
        belt_speed_mm_s: float,
        travel_fn=None,
    ) -> Dispatch | None:
        """Called every control tick. Returns a dispatch when the free arm is given a new pick."""
        for t in tracks:  # a confirmed piece past its deadline that we never issued is a miss
            if t.status == "confirmed" and t.id not in self.issued and t.deadline_s < now_s:
                self.missed.setdefault(t.id, now_s)
        if arm.busy:
            return None
        cands = [
            t for t in tracks
            if t.status == "confirmed" and t.id not in self.issued
            and t.id not in self.missed and t.confidence >= self.cfg["control"]["min_confidence"]
        ]  # fmt: skip
        cands.sort(key=lambda t: t.deadline_s)  # earliest deadline first
        for t in cands:
            if t.touching and fingers_collide(t, tracks, self.cfg):
                self._skip(t.id, "touching_collision")
                continue
            plan = plan_intercept(
                t, now_s, encoder_now, arm.free_at_s, arm.rest_mm, arm.yaw_deg,
                self.p, self.cfg, belt_speed_mm_s, travel_fn,
            )  # fmt: skip
            if not plan.feasible:
                self._skip(t.id, plan.reason)
                continue
            second = self._second(
                t, plan, cands, tracks, now_s, encoder_now, belt_speed_mm_s, travel_fn
            )
            return self._dispatch(arm, plan, second, now_s, belt_speed_mm_s)
        return self._last_resort(cands, now_s, encoder_now, arm, belt_speed_mm_s, travel_fn)

    def _last_resort(self, cands, now_s, encoder_now, arm, v, travel_fn) -> Dispatch | None:
        """Optional: a finger-blocked piece that is about to leave the zone is tried anyway rather
        than lost. The command is flagged collision_risk so the outcome is never mistaken for a
        clean pick."""
        c = self.cfg["control"]
        if not c["last_resort_blocked"]:
            return None
        for t in cands:
            if self.skip_reasons.get(t.id) != "touching_collision":
                continue
            x_now = (
                t.x_mm + (encoder_now - t.ref_encoder) * self.cfg["belt"]["mm_per_encoder_count"]
            )
            if self.p.zone_x_mm[1] - x_now > c["last_resort_mm"]:
                continue  # by distance, not time: a slowed belt must not postpone it forever
            plan = plan_intercept(
                t, now_s, encoder_now, arm.free_at_s, arm.rest_mm, arm.yaw_deg,
                self.p, self.cfg, v, travel_fn,
            )  # fmt: skip
            if plan.feasible:
                return self._dispatch(arm, plan, None, now_s, v, collision_risk=True)
        return None

    def _second(self, first, plan, cands, tracks, now_s, encoder_now, v, travel_fn):
        """Optional dual pick: a nearby piece needing a similar yaw, reachable on the same trip."""
        r = self.cfg["robot"]
        if not r["dual_pick"]:
            return None
        travel = travel_fn or (lambda dt: v * dt)
        for o in cands:
            if o.id == first.id:
                continue
            gap = float(np.hypot(o.x_mm - first.x_mm, o.y_mm - first.y_mm))
            yaw_o = (o.angle_deg + r["gripper_yaw_offset_deg"]) % 180.0
            if gap > r["dual_pick_radius_mm"]:
                continue
            if wrist_rotation_deg(plan.gripper_yaw_deg, yaw_o, True) > r["dual_pick_max_yaw_deg"]:
                continue
            if o.touching and fingers_collide(o, [t for t in tracks if t.id != first.id], self.cfg):
                continue
            extra = move_time_s(gap, self.p.vmax_mm_s, self.p.amax_mm_s2) + self.p.grip_s
            t2 = plan.grip_at_t_s + self.p.grip_s + (extra - self.p.grip_s)  # arrival at the second
            x2 = o.x_mm + (encoder_now - o.ref_encoder) * self.cfg["belt"]["mm_per_encoder_count"]
            x2 += travel(t2 - now_s)
            inside = self.p.zone_x_mm[0] <= x2 <= self.p.zone_x_mm[1]
            near = (
                np.hypot(x2 - self.p.base_mm[0], o.y_mm - self.p.base_mm[1])
                <= self.p.reach_radius_mm
            )
            if inside and near:
                return InterceptPlan(
                    o.id, True, pick_x_mm=x2, pick_y_mm=o.y_mm, gripper_yaw_deg=yaw_o,
                    grip_at_t_s=t2, depart_t_s=plan.depart_t_s, move_s=extra - self.p.grip_s,
                )  # fmt: skip
        return None

    def _dispatch(self, arm, plan, second, now_s, v, collision_risk: bool = False) -> Dispatch:
        cmd = plan.to_command(self.p)
        extra = None
        if second:
            extra = SecondPick(second.track_id, second.move_s + self.p.grip_s)
        rec = self.log.write(cmd, now_s, [second.track_id] if second else [], collision_risk)
        arm.begin(plan, v, extra)
        self.issued.add(plan.track_id)
        if second:
            self.issued.add(second.track_id)
        return Dispatch(plan, cmd, rec, second)
