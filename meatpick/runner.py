"""The closed loop: world -> camera -> detector -> tracker -> scheduler -> arm -> world.

Perception sees only rendered pixels and the encoder count. The scheduler and arm see only tracks.
The world's truth is used in exactly two places, both physics rather than knowledge: removing the
piece the gripper really closes on, and scoring afterwards.

A grip is resolved against reality. The arm follows the live belt encoder, so the gripper lands
where the *track* says the piece is *now*; the nearest real piece within ``grip_tolerance_mm`` is
the one picked. A track that is off by more than that (or a duplicate of a piece already taken)
closes on nothing and is counted as an empty grip.
"""

from __future__ import annotations

import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from meatpick.config import make_rng
from meatpick.control.arm_model import Arm
from meatpick.control.belt_controller import BeltController
from meatpick.control.scheduler import CommandLog, Scheduler
from meatpick.perception.calibration import Calibration
from meatpick.perception.factory import Perceiver
from meatpick.perception.tracker import Tracker, track_position_at
from meatpick.sim.calibration_target import render_calibration_set
from meatpick.sim.camera import Camera
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World

LINK_MM = 30.0  # a track this close to a real piece is "the same piece" when scoring


def run_closed_loop(
    cfg: dict,
    minutes: float = 5.0,
    out_dir: str | Path | None = None,
    detector: str | None = None,
    progress: bool = False,
) -> dict:
    cfg = copy.deepcopy(cfg)
    rng = make_rng(cfg["seed"])
    cam = Camera(cfg, rng)
    tilted, flat, _ = render_calibration_set(cam, cfg, rng)
    cal = Calibration.calibrate(tilted, flat)
    perceiver, tracker = Perceiver(cfg, cal, prefer=detector), Tracker(cfg, cal)
    world = World(cfg, TextureBank(cfg, rng), rng)

    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        for name in ("commands.jsonl", "events.jsonl"):
            (out / name).unlink(missing_ok=True)
    sched = Scheduler(cfg, CommandLog(out / "commands.jsonl" if out else None, cfg))
    arm, belt = Arm(sched.p), BeltController(cfg)
    events: list[dict] = []

    def log(kind: str, **kw) -> None:
        events.append({"t_s": round(world.t_s, 3), "kind": kind, **kw})

    mm = cfg["belt"]["mm_per_encoder_count"]
    tol = cfg["control"]["grip_tolerance_mm"]
    dt = cfg["world"]["dt_s"]
    pending: dict[int, tuple[float, float]] = {}  # track id -> planned pick point
    piece_tracks: dict[int, set[int]] = defaultdict(set)  # world piece id -> track ids seen on it
    grip_err, speeds, perceive_ms, frames = [], [], [], 0
    empty_grips = stale_grips = stopped_steps = 0
    seen_missed: set[int] = set()
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=int(minutes * 60 / dt), desc="closed loop", unit="step", mininterval=2)
    t_start = time.perf_counter()

    for step in range(int(minutes * 60 / dt)):
        world.step()
        if bar and step % 100 == 0:
            bar.update(100)
        got = cam.maybe_capture(world)
        if got is not None:
            frame, _truth = got  # truth is dropped here: only pixels, encoder and time go on
            t0 = time.perf_counter()
            tracker.update(frame, perceiver.perceive(frame))
            perceive_ms.append((time.perf_counter() - t0) * 1000.0)
            frames += 1
            for tr in tracker.tracks:  # scoring only: which real piece does each track sit on?
                tx, ty = track_position_at(tr, world.belt.encoder, mm)
                for p in world.pieces:
                    if p.state == "on_belt" and np.hypot(p.x_mm - tx, p.y_mm - ty) < LINK_MM:
                        piece_tracks[p.id].add(tr.id)
        tracks = tracker.tracks

        v_mm_s = world.belt.speed_mps * 1000.0
        skip = sched.issued | set(sched.missed) | sched.unpickable(tracks)
        world.belt.command(
            belt.update(world.t_s, dt, world.belt.encoder, tracks, skip, arm, world.belt.speed_mps)
        )
        d = sched.step(world.t_s, world.belt.encoder, tracks, arm, v_mm_s, belt.predict_travel)
        if d:
            for tid in d.track_ids:
                tracker.mark_assigned(tid)
            pending[d.plan.track_id] = (d.plan.pick_x_mm, d.plan.pick_y_mm)
            if d.second_plan:
                pending[d.second_plan.track_id] = (d.second_plan.pick_x_mm, d.second_plan.pick_y_mm)
            log("dispatch", track_ids=d.track_ids, belt_mps=round(world.belt.speed_mps, 3),
                move_s=round(d.plan.move_s, 3))  # fmt: skip

        for ev in arm.advance(world.t_s):
            if ev.kind == "grip":
                planned = pending.pop(ev.track_id, None)
                tr = tracker.get(ev.track_id)
                if tr is not None:  # the arm follows the live encoder, not the stale plan
                    gx, gy = track_position_at(tr, world.belt.encoder, mm)
                elif planned is not None:
                    gx, gy = planned
                    stale_grips += 1
                else:
                    continue
                cands = [p for p in world.pieces if p.state == "on_belt"]
                best = min(cands, key=lambda p: np.hypot(p.x_mm - gx, p.y_mm - gy), default=None)
                e = float(np.hypot(best.x_mm - gx, best.y_mm - gy)) if best else float("inf")
                if best is not None and e <= tol:
                    world.pick(best.id)
                    grip_err.append(e)
                    log("grip", track_id=ev.track_id, piece_id=best.id, error_mm=round(e, 2))
                else:
                    empty_grips += 1
                    log("empty_grip", track_id=ev.track_id, nearest_error_mm=round(e, 1),
                        track_xy_mm=[round(gx, 1), round(gy, 1)],
                        nearest_piece=None if best is None else
                        [best.id, round(best.x_mm, 1), round(best.y_mm, 1)],
                        track_hits=None if tr is None else tr.hits)  # fmt: skip
                tracker.mark_picked(ev.track_id)
            elif ev.kind == "release":
                log("release", track_id=ev.track_id)
        for p in world.pieces:
            if p.state == "missed" and p.id not in seen_missed:
                seen_missed.add(p.id)
                log("miss", piece_id=p.id)
        speeds.append(world.belt.speed_mps)
        stopped_steps += world.belt.speed_mps < 0.005
    wall_s = time.perf_counter() - t_start
    if bar:
        bar.close()

    # --- score ---------------------------------------------------------------------------
    blocked = {i for i, r in sched.skip_reasons.items() if r == "touching_collision"}
    cause: Counter = Counter()
    for pid in seen_missed:
        tids = piece_tracks.get(pid, set())
        if not tids:
            cause["never_tracked"] += 1
        elif tids & blocked:
            cause["finger_blocked"] += 1
        else:
            cause["timing_or_other"] += 1
    done = world.n_picked + world.n_missed
    lat = np.array(perceive_ms) if perceive_ms else np.zeros(1)
    metrics = {
        "adaptive": cfg["control"]["adaptive_belt"],
        "detector": perceiver.description,
        "seed": cfg["seed"],
        "rate_per_min": cfg["arrivals"]["rate_per_min"],
        "minutes": minutes,
        "spawned": world.n_spawned,
        "picked": world.n_picked,
        "missed": world.n_missed,
        "miss_rate": round(world.n_missed / max(done, 1), 4),
        "picks_per_min": round(world.n_picked / minutes, 2),
        "miss_causes": dict(cause),
        "empty_grips": empty_grips,
        "stale_plan_grips": stale_grips,
        "grip_error_mm_mean": round(float(np.mean(grip_err)), 3) if grip_err else None,
        "grip_error_mm_max": round(float(np.max(grip_err)), 3) if grip_err else None,
        "commands": len(sched.log.records),
        "risky_picks": sum(1 for r in sched.log.records if r.get("collision_risk")),
        "skips": dict(Counter(sched.skip_reasons.values())),
        "mean_belt_mps": round(float(np.mean(speeds)), 4),
        "belt_stopped_fraction": round(stopped_steps / max(len(speeds), 1), 4),
        "belt_stops_commanded": belt.stop_count,
        "frames": frames,
        "perception_ms_mean": round(float(lat.mean()), 1),
        "perception_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "wall_s": round(wall_s, 1),
        "realtime_factor": round(minutes * 60 / wall_s, 2),
    }
    if out:
        (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
