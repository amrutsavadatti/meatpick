import json

import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.control.arm_model import Arm, ArmState, robot_to_belt
from meatpick.control.scheduler import (
    CommandLog,
    Scheduler,
    command_record,
    fingers_collide,
    read_commands,
    validate_command_record,
)
from meatpick.types import PickCommand, Track

V = 250.0  # belt speed, mm/s


@pytest.fixture
def cfg():
    return load_config()


def trk(i, x, y=300.0, angle=0.0, status="confirmed", touching=False, width=85.0, length=140.0,
        conf=0.9, deadline=None):  # fmt: skip
    return Track(
        i, x, y, 1000, angle, 1, conf, (1000.0 - x) / V if deadline is None else deadline,
        status, touching=touching, length_mm=length, width_mm=width,
    )  # fmt: skip


def run_step(sched, tracks, arm=None, now=0.0, cfg=None):
    arm = arm or Arm(sched.p)
    return sched.step(now, 1000, tracks, arm, V), arm


def test_earliest_deadline_first(cfg):
    s = Scheduler(cfg)
    d, _ = run_step(s, [trk(1, 650), trk(2, 720), trk(3, 680)])
    assert d.plan.track_id == 2  # most downstream = earliest deadline


def test_only_confirmed_tracks_are_scheduled(cfg):
    s = Scheduler(cfg)
    tracks = [
        trk(i, 700 + i, status=st)
        for i, st in enumerate(["tentative", "assigned", "picked", "missed"])
    ]
    assert run_step(s, tracks)[0] is None


def test_a_track_is_never_issued_twice_even_if_still_reported(cfg):
    s = Scheduler(cfg)
    tracks = [trk(1, 700)]
    first, arm = run_step(s, tracks)
    assert first.plan.track_id == 1
    arm.advance(arm.free_at_s + 0.01)  # cycle done, arm idle again
    again = s.step(arm.free_at_s, 1000, tracks, arm, V)
    assert again is None  # the tracker may lag, but we never re-send a piece we dispatched


def test_unreachable_pieces_get_no_command(cfg):
    s = Scheduler(cfg)
    off_belt = trk(1, 800, y=-400)  # far outside the arm's radius
    too_late = trk(2, 990, angle=100.0)
    behind = trk(3, 1100)  # already past the zone
    d, _ = run_step(s, [off_belt, too_late, behind])
    assert d is None and s.log.records == []
    assert s.skip_reasons[1] == "outside_reach" and s.skip_reasons[2] == "too_late"


def test_a_busy_arm_gets_nothing(cfg):
    s = Scheduler(cfg)
    _, arm = run_step(s, [trk(1, 700)])
    assert s.step(0.1, 1000, [trk(2, 700)], arm, V) is None


def test_an_unreachable_first_choice_does_not_block_the_next(cfg):
    s = Scheduler(cfg)
    d, _ = run_step(s, [trk(1, 995, angle=100.0), trk(2, 700)])  # #1 has the earlier deadline
    assert d.plan.track_id == 2


def test_low_confidence_tracks_are_not_scheduled(cfg):
    s = Scheduler(cfg)
    assert run_step(s, [trk(1, 700, conf=0.1)])[0] is None


def test_a_confirmed_piece_past_its_deadline_is_recorded_as_a_miss_once(cfg):
    s = Scheduler(cfg)
    late = trk(1, 700, deadline=-1.0)
    run_step(s, [late], now=0.0)
    run_step(s, [late], now=0.5)
    assert list(s.missed) == [1] and s.log.records == []


# --- touching / finger collisions -------------------------------------------------------
def test_a_neighbour_beside_the_long_edge_blocks_the_fingers_but_one_at_the_end_does_not(cfg):
    p = trk(1, 700.0, 300.0, angle=0.0, touching=True)  # long axis along belt x
    beside = trk(2, 700.0, 300.0 + 85.0, angle=0.0, touching=True)
    at_end = trk(3, 700.0 + 140.0, 300.0, angle=0.0, touching=True)
    assert fingers_collide(p, [p, beside], cfg)
    assert not fingers_collide(p, [p, at_end], cfg)


def test_touching_pieces_are_skipped_only_when_the_fingers_would_collide(cfg):
    blocked = trk(1, 720.0, 300.0, touching=True)
    neighbour = trk(2, 720.0, 385.0, touching=True)
    clear = trk(3, 690.0, 150.0, touching=True)  # flagged touching, nothing near its fingers
    end_a, end_b = trk(4, 700.0, 520.0, touching=True), trk(5, 840.0, 520.0, touching=True)
    s = Scheduler(cfg)
    d, _ = run_step(s, [blocked, neighbour, clear])
    assert s.skip_reasons.get(1) == "touching_collision" and d.plan.track_id in (2, 3)
    s2 = Scheduler(cfg)
    d2, _ = run_step(s2, [end_a, end_b])  # end to end: fingers are clear
    assert d2 is not None and 4 not in s2.skip_reasons and 5 not in s2.skip_reasons


def test_an_unflagged_piece_is_never_blocked(cfg):
    s = Scheduler(cfg)
    d, _ = run_step(s, [trk(1, 720.0, 300.0), trk(2, 720.0, 385.0, touching=True)])
    assert d is not None and 1 not in s.skip_reasons


# --- the command interface --------------------------------------------------------------
def good_record(cfg):
    cmd = PickCommand(7, 10.0, -20.0, 130.0, 4000, 1.5)
    return command_record(cmd, 0.5, [])


def test_a_valid_record_passes_and_bad_ones_are_named(cfg):
    ok = good_record(cfg)
    assert validate_command_record(ok, cfg) == []
    bad = {
        "yaw 180": {"gripper_yaw_deg": 180.0},
        "wrong place": {"place": "CONVEYOR"},
        "negative id": {"track_id": -1},
        "bool id": {"track_id": True},
        "float encoder": {"grip_at_encoder": 12.5},
        "before issue": {"grip_at_t_s": 0.1},
        "outside reach": {"pick_x_mm": 600.0, "pick_y_mm": 0.0},
        "bad also_pick": {"also_pick": ["a"]},
    }
    for name, change in bad.items():
        assert validate_command_record({**ok, **change}, cfg), name
    missing = dict(ok)
    del missing["place"]
    assert validate_command_record(missing, cfg)


def test_the_log_writes_valid_json_lines_and_refuses_invalid_commands(cfg, tmp_path):
    log = CommandLog(tmp_path / "commands.jsonl", cfg)
    log.write(PickCommand(1, 5.0, 5.0, 90.0, 100, 1.0), 0.0, [])
    log.write(PickCommand(2, -5.0, 5.0, 10.0, 200, 2.0), 0.5, [3])
    rows = read_commands(tmp_path / "commands.jsonl")
    assert [r["track_id"] for r in rows] == [1, 2] and rows[1]["also_pick"] == [3]
    assert all(validate_command_record(r, cfg) == [] for r in rows)
    with pytest.raises(ValueError):
        log.write(PickCommand(3, 900.0, 0.0, 10.0, 1, 3.0), 1.0, [])  # beyond reach
    assert len(read_commands(tmp_path / "commands.jsonl")) == 2  # nothing bad was written


def test_dispatched_commands_validate_and_use_the_robot_frame(cfg, tmp_path):
    s = Scheduler(cfg, CommandLog(tmp_path / "c.jsonl", cfg))
    d, arm = run_step(s, [trk(1, 700.0, 250.0)])
    assert validate_command_record(d.record, cfg) == []
    bx, by = robot_to_belt(s.p, d.command.pick_x_mm, d.command.pick_y_mm)
    assert (bx, by) == pytest.approx((d.plan.pick_x_mm, d.plan.pick_y_mm))
    assert json.loads((tmp_path / "c.jsonl").read_text().splitlines()[0])["track_id"] == 1


# --- dual pick ---------------------------------------------------------------------------
def dual_cfg(on=True):
    return load_config(overrides=[f"robot.dual_pick={'true' if on else 'false'}"])


def test_dual_pick_takes_a_close_piece_with_a_similar_yaw_on_the_same_trip():
    cfg = dual_cfg()
    s = Scheduler(cfg)
    tracks = [trk(1, 720.0, 300.0, angle=10.0), trk(2, 700.0, 350.0, angle=20.0)]
    d, arm = run_step(s, tracks)
    assert d.second_plan is not None and d.track_ids == [1, 2] and d.record["also_pick"] == [2]
    assert (
        arm.free_at_s
        > Scheduler(dual_cfg(False)).step(0.0, 1000, tracks, Arm(s.p), V).plan.grip_at_t_s
    )
    events = arm.advance(arm.free_at_s + 0.01)
    assert [(e.kind, e.track_id) for e in events if e.kind != "idle"] == [
        ("grip", 1), ("grip", 2), ("release", 1), ("release", 2),
    ]  # fmt: skip
    assert s.issued == {1, 2}


def test_dual_pick_is_off_by_default_and_needs_proximity_and_similar_yaw(cfg):
    tracks = [trk(1, 720.0, 300.0, angle=10.0), trk(2, 700.0, 350.0, angle=20.0)]
    assert run_step(Scheduler(cfg), tracks)[0].second_plan is None  # default config: single picks
    on = dual_cfg()
    far = [trk(1, 720.0, 300.0, angle=10.0), trk(2, 700.0, 480.0, angle=15.0)]
    assert run_step(Scheduler(on), far)[0].second_plan is None
    twisted = [trk(1, 720.0, 300.0, angle=10.0), trk(2, 700.0, 350.0, angle=70.0)]
    assert run_step(Scheduler(on), twisted)[0].second_plan is None


# --- a short closed loop with perfect tracks (a preview of #14) ---------------------------
def test_closed_loop_with_perfect_tracks_never_double_issues_or_sends_invalid_commands(
    cfg, tmp_path
):
    from meatpick.sim.textures import TextureBank
    from meatpick.sim.world import World, touching_flags

    rng = make_rng(11)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    sched = Scheduler(cfg, CommandLog(tmp_path / "commands.jsonl", cfg))
    arm = Arm(sched.p)
    lo, hi = cfg["robot"]["reach_zone_x_mm"]
    flags, issued_world_ids, grip_errors, picked, tried = {}, [], [], 0, 0
    pending = {}  # track id -> planned (x, y, t) per dispatch
    for step in range(9000):  # 90 s
        world.step()
        if step % 10 == 0:
            truths = [g for g in world.ground_truth() if 100 < g.x_mm < hi + 100]
            flags = touching_flags(truths, cfg["robot"]["touching_clearance_mm"]) if truths else {}
        v = world.belt.speed_mps * 1000.0
        tracks = [
            Track(p.id, p.x_mm, p.y_mm, world.belt.encoder, p.theta_deg, 1, 1.0,
                  world.t_s + (hi - p.x_mm) / v, "confirmed", touching=flags.get(p.id, False),
                  length_mm=p.sprite.length_mm, width_mm=p.sprite.width_mm)
            for p in world.pieces if p.state == "on_belt" and 100 < p.x_mm < hi + 50
        ]  # fmt: skip
        d = sched.step(world.t_s, world.belt.encoder, tracks, arm, v)
        if d:
            issued_world_ids += d.track_ids
            pending[d.plan.track_id] = (d.plan.pick_x_mm, d.plan.pick_y_mm)
        for ev in arm.advance(world.t_s):
            if ev.kind == "grip" and ev.track_id in pending:
                tried += 1
                px, py = pending.pop(ev.track_id)
                near = [q for q in world.pieces if q.id == ev.track_id and q.state == "on_belt"]
                if near and np.hypot(near[0].x_mm - px, near[0].y_mm - py) < 3.0:
                    grip_errors.append(float(np.hypot(near[0].x_mm - px, near[0].y_mm - py)))
                    world.pick(ev.track_id)
                    picked += 1
    assert len(issued_world_ids) == len(set(issued_world_ids))  # no piece sent twice
    rows = read_commands(tmp_path / "commands.jsonl")
    assert rows and all(validate_command_record(r, cfg) == [] for r in rows)
    assert tried == picked and max(grip_errors) < 3.0  # the arm was where the piece really was
    assert (
        picked >= 0.6 * world.n_spawned * (90.0 - 10.0) / 90.0
    )  # sanity floor, not the #17 result
    assert arm.state in set(ArmState)


# --- last-resort picks (experimental, off by default) -----------------------------------
def test_last_resort_picks_a_blocked_piece_only_near_the_end_of_the_zone_and_flags_it():
    on = load_config(overrides=["control.last_resort_blocked=true"])
    blocked = trk(1, 820.0, 300.0, touching=True)  # inside the last 200 mm, still reachable
    neighbour = trk(2, 820.0, 385.0, touching=True)
    far = trk(3, 640.0, 300.0, touching=True)
    far_n = trk(4, 640.0, 385.0, touching=True)
    s = Scheduler(on)
    d, _ = run_step(s, [far, far_n])
    assert d is None  # blocked and nowhere near the end of the zone: still skipped
    s = Scheduler(on)
    d, _ = run_step(s, [blocked, neighbour])
    assert d is not None and d.record["collision_risk"] is True
    assert validate_command_record(d.record, on) == []


def test_blocked_pieces_are_never_tried_unless_last_resort_is_enabled(cfg):
    s = Scheduler(cfg)
    d, _ = run_step(s, [trk(1, 940.0, 300.0, touching=True), trk(2, 940.0, 385.0, touching=True)])
    assert d is None
