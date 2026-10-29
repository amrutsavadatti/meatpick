"""A lightweight closed loop that uses TRUE piece positions as the tracks.

No camera, detector or tracker: it isolates the scheduler, arm model and belt controller, so
their behaviour (misses at a fixed vs adaptive belt, grip accuracy) can be measured fast. The real
loop with perception is the runner (#14); numbers here are an upper bound on what it can achieve.
"""

from __future__ import annotations

import copy

import numpy as np

from meatpick.config import make_rng
from meatpick.control.arm_model import Arm
from meatpick.control.belt_controller import BeltController
from meatpick.control.scheduler import Scheduler
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World, touching_flags
from meatpick.types import Track


def run_truth_loop(
    cfg: dict, minutes: float = 5.0, seed: int = 0, adaptive: bool | None = None,
    log_path: str | None = None,
) -> dict:  # fmt: skip
    cfg = copy.deepcopy(cfg)
    if adaptive is not None:
        cfg["control"]["adaptive_belt"] = adaptive
    rng = make_rng(seed)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    from meatpick.control.scheduler import CommandLog

    sched = Scheduler(cfg, CommandLog(log_path, cfg) if log_path else None)
    arm, belt = Arm(sched.p), BeltController(cfg)
    lo, hi = cfg["robot"]["reach_zone_x_mm"]
    clearance = cfg["robot"]["touching_clearance_mm"]
    dt = cfg["world"]["dt_s"]
    flags: dict = {}
    pending: dict = {}  # track id -> planned pick point
    picked = 0
    grip_err, speeds, dispatch_t = [], [], []
    stopped_s = 0.0
    ever_blocked: set[int] = set()
    missed_ids: set[int] = set()
    for step in range(int(minutes * 60 / dt)):
        world.step()
        missed_ids |= {q.id for q in world.pieces if q.state == "missed"}
        if step % 10 == 0:  # touching flags change slowly: refresh every 0.1 s
            near = [g for g in world.ground_truth() if 100 < g.x_mm < hi + 100]
            flags = touching_flags(near, clearance) if near else {}
        v = world.belt.speed_mps * 1000.0
        tracks = [
            Track(
                p.id, p.x_mm, p.y_mm, world.belt.encoder, p.theta_deg, 1, 1.0,
                world.t_s + (hi - p.x_mm) / max(v, 1.0), "confirmed",
                touching=flags.get(p.id, False), length_mm=p.sprite.length_mm,
                width_mm=p.sprite.width_mm,
            )
            for p in world.pieces
            if p.state == "on_belt" and 100 < p.x_mm < hi + 50
        ]  # fmt: skip
        world.belt.command(
            belt.update(world.t_s, dt, world.belt.encoder, tracks,
                        sched.issued | set(sched.missed) | sched.unpickable(tracks), arm,
                        world.belt.speed_mps)
        )  # fmt: skip
        d = sched.step(world.t_s, world.belt.encoder, tracks, arm, v, belt.predict_travel)
        ever_blocked |= {i for i, r in sched.skip_reasons.items() if r == "touching_collision"}
        if d:
            pending[d.plan.track_id] = (d.plan.pick_x_mm, d.plan.pick_y_mm)
            dispatch_t.append(world.t_s)
        for ev in arm.advance(world.t_s):
            if ev.kind == "grip" and ev.track_id in pending:
                px, py = pending.pop(ev.track_id)
                piece = next((q for q in world.pieces if q.id == ev.track_id), None)
                if piece is not None and piece.state == "on_belt":
                    grip_err.append(float(np.hypot(piece.x_mm - px, piece.y_mm - py)))
                    world.pick(ev.track_id)
                    picked += 1
        speeds.append(world.belt.speed_mps)
        stopped_s += dt if world.belt.speed_mps < 0.005 else 0.0
    done = world.n_picked + world.n_missed  # pieces whose fate is decided
    return {
        "adaptive": cfg["control"]["adaptive_belt"],
        "seed": seed,
        "minutes": minutes,
        "spawned": world.n_spawned,
        "picked": world.n_picked,
        "missed": world.n_missed,
        "miss_rate": world.n_missed / max(done, 1),
        "missed_after_finger_block": len(missed_ids & ever_blocked),
        "missed_other": len(missed_ids - ever_blocked),
        "picks_per_min": world.n_picked / minutes,
        "mean_belt_mps": float(np.mean(speeds)),
        "min_belt_mps": float(np.min(speeds)),
        "belt_stopped_fraction": stopped_s / (minutes * 60),
        "belt_stops_commanded": belt.stop_count,
        "grip_error_mm_mean": float(np.mean(grip_err)) if grip_err else float("nan"),
        "grip_error_mm_max": float(np.max(grip_err)) if grip_err else float("nan"),
        "skips": dict(__import__("collections").Counter(sched.skip_reasons.values())),
        "commands": len(sched.log.records),
        "risky_picks": sum(1 for r in sched.log.records if r.get("collision_risk")),
    }
