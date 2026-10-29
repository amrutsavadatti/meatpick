import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.control.arm_model import (
    Arm,
    ArmParams,
    ArmState,
    belt_to_robot,
    move_time_s,
    pick_move_time_s,
    robot_to_belt,
    wrist_rotation_deg,
)
from meatpick.control.intercept import plan_intercept
from meatpick.types import Track

V, A = 2000.0, 10000.0  # PRD defaults in mm/s and mm/s^2


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def p(cfg):
    return ArmParams.from_cfg(cfg)


def track(x=700.0, y=300.0, angle=40.0, enc=1000, tid=1):
    return Track(tid, x, y, enc, angle, 1, 1.0, 99.0, "confirmed")


def plan(cfg, p, tr, now=0.0, enc=1000, free=0.0, rest=None, yaw=0.0, v=250.0, travel=None):
    return plan_intercept(tr, now, enc, free, rest or p.bin_mm, yaw, p, cfg, v, travel)


# --- move time ---------------------------------------------------------------------------
def test_move_time_matches_the_closed_form_in_both_regimes():
    assert move_time_s(0, V, A) == 0.0
    assert move_time_s(100, V, A) == pytest.approx(2 * np.sqrt(100 / A))  # never reaches vmax
    assert move_time_s(1000, V, A) == pytest.approx(1000 / V + V / A)  # cruises


def test_the_prd_cycle_budget_a_400mm_move_takes_0_4_s():
    assert move_time_s(400, V, A) == pytest.approx(0.4)
    assert 0.4 + 0.15 + 0.4 + 0.10 == pytest.approx(1.05)  # leaves ~0.15 s of the 1.2 s budget


def test_move_time_is_continuous_at_the_regime_boundary_and_monotonic():
    edge = V**2 / A
    assert move_time_s(edge - 1e-6, V, A) == pytest.approx(move_time_s(edge + 1e-6, V, A), abs=1e-6)
    d = np.linspace(0, 2000, 200)
    t = [move_time_s(x, V, A) for x in d]
    assert all(b >= a for a, b in zip(t, t[1:], strict=False))


def test_faster_arm_is_never_slower_and_short_moves_ignore_vmax():
    assert move_time_s(800, 3000, A) < move_time_s(800, V, A)
    assert move_time_s(100, 5000, A) == pytest.approx(move_time_s(100, V, A))


def test_wrist_penalty_applies_only_above_the_threshold(p):
    base = pick_move_time_s(300, 0.0, p)
    assert pick_move_time_s(300, 90.0, p) == pytest.approx(base)
    assert pick_move_time_s(300, 90.1, p) == pytest.approx(base + p.wrist_penalty_s)


def test_wrist_rotation_can_use_gripper_symmetry():
    assert wrist_rotation_deg(10, 170) == 160
    assert wrist_rotation_deg(10, 170, symmetric=True) == pytest.approx(20)


def test_frame_conversion_round_trips(p):
    assert robot_to_belt(p, *belt_to_robot(p, 812.0, 77.0)) == pytest.approx((812.0, 77.0))


# --- intercept ---------------------------------------------------------------------------
def test_on_a_stopped_belt_the_arm_goes_straight_to_the_piece(cfg, p):
    pl = plan(cfg, p, track(x=800.0, y=250.0), v=0.0)
    d = np.hypot(800 - p.bin_mm[0], 250 - p.bin_mm[1])
    assert pl.feasible and pl.pick_x_mm == pytest.approx(800.0)
    assert pl.grip_at_t_s == pytest.approx(pick_move_time_s(d, pl.wrist_rotation_deg, p))


def test_moving_belt_solution_is_self_consistent_and_converges_fast(cfg, p):
    tr = track(x=700.0, y=200.0, angle=100.0)  # yaw 190 -> 10, small rotation
    pl = plan(cfg, p, tr, yaw=10.0)
    assert pl.feasible and pl.converged and pl.iterations <= 3
    d = np.hypot(pl.pick_x_mm - p.bin_mm[0], pl.pick_y_mm - p.bin_mm[1])
    assert pl.grip_at_t_s - pl.depart_t_s == pytest.approx(
        pick_move_time_s(d, pl.wrist_rotation_deg, p), abs=1e-3
    )
    assert pl.pick_x_mm == pytest.approx(700.0 + 250.0 * pl.grip_at_t_s, abs=1e-6)


def test_the_arm_waits_for_a_piece_that_has_not_reached_the_zone(cfg, p):
    pl = plan(cfg, p, track(x=300.0))  # 300 mm upstream of the zone: 1.2 s away
    assert pl.feasible and pl.grip_at_t_s == pytest.approx(1.2)
    assert pl.pick_x_mm == pytest.approx(p.zone_x_mm[0])


def test_a_piece_the_arm_cannot_reach_in_time_is_rejected(cfg, p):
    late = plan(
        cfg, p, track(x=990.0, angle=100.0), yaw=10.0
    )  # exits the zone before the arm arrives
    assert not late.feasible and late.reason == "too_late"
    assert not plan(cfg, p, track(x=1100.0)).feasible


def test_a_busy_arm_delays_the_departure(cfg, p):
    free = plan(cfg, p, track(x=800.0), v=0.0)
    busy = plan(cfg, p, track(x=800.0), free=0.7, v=0.0)
    assert busy.depart_t_s == pytest.approx(0.7)
    assert busy.grip_at_t_s == pytest.approx(free.grip_at_t_s + 0.7)


def test_pieces_outside_the_arms_radius_are_rejected(cfg, p):
    pl = plan(cfg, p, track(x=800.0, y=-300.0), v=0.0)  # far off the belt
    assert not pl.feasible and pl.reason == "outside_reach"


def test_a_stopped_belt_never_brings_an_upstream_piece_into_the_zone(cfg, p):
    pl = plan(cfg, p, track(x=300.0), v=0.0)
    assert not pl.feasible and pl.reason == "belt_stopped"


def test_gripper_yaw_is_the_long_axis_plus_90(cfg, p):
    assert plan(cfg, p, track(angle=40.0), v=0.0).gripper_yaw_deg == pytest.approx(130.0)
    assert plan(cfg, p, track(angle=120.0), v=0.0).gripper_yaw_deg == pytest.approx(30.0)  # mod 180


def test_the_encoder_count_at_grip_follows_belt_travel(cfg, p):
    pl = plan(
        cfg, p, track(x=300.0, enc=1000), enc=1000
    )  # grips at t=1.2 s -> 300 mm -> 3000 counts
    assert pl.grip_at_encoder == 1000 + 3000


def test_track_position_is_advanced_to_the_current_encoder(cfg, p):
    # same piece described at an earlier encoder count: 1000 counts = 100 mm further now
    a = plan(cfg, p, track(x=600.0, enc=0), enc=1000, v=0.0)
    b = plan(cfg, p, track(x=700.0, enc=1000), enc=1000, v=0.0)
    assert a.pick_x_mm == pytest.approx(b.pick_x_mm)


def test_command_is_in_the_robot_frame(cfg, p):
    pl = plan(cfg, p, track(x=800.0, y=250.0), v=0.0)
    cmd = pl.to_command(p)
    assert (cmd.pick_x_mm, cmd.pick_y_mm) == pytest.approx((0.0, -50.0))
    assert cmd.place == "BIN" and cmd.grip_at_t_s == pl.grip_at_t_s


def test_belt_that_is_slowing_needs_a_predictor_that_knows_it(cfg, p):
    """Constant-speed prediction is wrong when the belt decelerates; a real predictor is exact."""
    v0, decel = 250.0, 500.0  # mm/s and mm/s^2, the PRD's 0.5 m/s^2 limit

    def true_travel(dt):
        t = min(dt, v0 / decel)  # stops after 0.5 s
        return v0 * t - 0.5 * decel * t * t

    tr = track(x=620.0, y=300.0)
    naive = plan(cfg, p, tr, v=v0)
    aware = plan(cfg, p, tr, v=v0, travel=true_travel)
    err_naive = abs(naive.pick_x_mm - (620.0 + true_travel(naive.grip_at_t_s)))
    err_aware = abs(aware.pick_x_mm - (620.0 + true_travel(aware.grip_at_t_s)))
    assert err_naive > 10.0 and err_aware < 1e-6


# --- state machine -----------------------------------------------------------------------
def make_plan(cfg, p, **kw):
    pl = plan(cfg, p, track(x=800.0, y=250.0), v=250.0, **kw)
    assert pl.feasible
    return pl


def test_cycle_runs_in_order_with_the_right_timestamps(cfg, p):
    arm = Arm(p)
    pl = make_plan(cfg, p)
    arm.begin(pl, 250.0)
    seen, events = [], []
    for t in np.arange(0.0, 3.0, 0.005):
        events += arm.advance(float(t))
        if not seen or seen[-1] != arm.state:
            seen.append(arm.state)
    assert seen == [
        ArmState.MOVING_TO_PICK, ArmState.GRIPPING, ArmState.MOVING_TO_BIN,
        ArmState.RELEASING, ArmState.IDLE,
    ]  # fmt: skip
    kinds = [(e.kind, round(e.t_s, 3)) for e in events]
    assert [k for k, _ in kinds] == ["grip", "release", "idle"]
    grip_t = pl.grip_at_t_s
    assert kinds[0][1] == pytest.approx(grip_t, abs=1e-3)
    # the move to the bin starts from where the belt carried the arm during the grip
    leave = (pl.pick_x_mm + 250.0 * p.grip_s, pl.pick_y_mm)
    t_bin = move_time_s(np.hypot(p.bin_mm[0] - leave[0], p.bin_mm[1] - leave[1]), V, A)
    assert kinds[1][1] == pytest.approx(grip_t + p.grip_s + t_bin, abs=1e-3)
    assert kinds[2][1] == pytest.approx(kinds[1][1] + p.release_s, abs=1e-3)
    assert arm.free_at_s == pytest.approx(kinds[2][1], abs=1e-3)


def test_a_new_cycle_cannot_start_while_busy_and_waits_for_free(cfg, p):
    arm = Arm(p)
    arm.begin(make_plan(cfg, p), 250.0)
    arm.advance(0.1)
    with pytest.raises(RuntimeError):
        arm.begin(make_plan(cfg, p), 250.0)
    arm.advance(arm.free_at_s + 0.01)
    assert not arm.busy
    nxt = plan(cfg, p, track(x=100.0, y=250.0), free=arm.free_at_s, v=250.0)  # still upstream
    assert nxt.feasible
    arm.begin(nxt, 250.0)  # fine once idle


def test_arm_ends_parked_at_the_bin_with_the_gripper_yaw_it_used(cfg, p):
    arm = Arm(p)
    pl = make_plan(cfg, p)
    arm.begin(pl, 250.0)
    assert arm.rest_mm == p.bin_mm and arm.yaw_deg == pl.gripper_yaw_deg


def test_a_full_400mm_cycle_costs_about_the_prd_budget(cfg, p):
    """Pick and bin 400 mm apart on a stopped belt: 0.4 + 0.15 + 0.4 + 0.1 (+ wrist penalty)."""
    arm = Arm(p)
    tr = track(x=800.0, y=300.0, angle=0.0)  # yaw 90: no wrist penalty from yaw 90
    pl = plan_intercept(tr, 0.0, 1000, 0.0, p.bin_mm, 90.0, p, cfg, 0.0)
    arm.begin(pl, 0.0)
    assert pl.move_s == pytest.approx(0.4) and arm.free_at_s == pytest.approx(1.05)


# --- in the simulator --------------------------------------------------------------------
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_planned_pick_point_matches_the_true_piece_within_2mm(seed, cfg, p):
    """PRD acceptance: perceive + track a real piece, plan the intercept, then compare the
    planned grip position with where the piece truly is at grip time."""
    from meatpick.perception.calibration import Calibration
    from meatpick.perception.factory import Perceiver
    from meatpick.perception.tracker import Tracker
    from meatpick.sim.calibration_target import render_calibration_set
    from meatpick.sim.camera import Camera
    from meatpick.sim.textures import TextureBank
    from meatpick.sim.world import World

    rng = make_rng(seed)
    cam = Camera(cfg, rng)
    tilted, flat, _ = render_calibration_set(cam, cfg, rng)
    cal = Calibration.calibrate(tilted, flat)
    perceiver, tracker = Perceiver(cfg, cal, "classical"), Tracker(cfg, cal)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    world.pieces = [world._make_piece(-150.0, float(rng.uniform(150, 450)))]
    plan_ = None
    for _ in range(4000):
        world.step()
        got = cam.maybe_capture(world)
        if got is not None:
            tracks = tracker.update(got[0], perceiver.perceive(got[0]))
            ready = [t for t in tracks if t.status == "confirmed" and 150 < t.x_mm < 280]
            if ready and plan_ is None:
                plan_ = plan_intercept(
                    ready[0], world.t_s, world.belt.encoder, 0.0, p.bin_mm, 0.0, p, cfg,
                    world.belt.speed_mps * 1000.0,
                )  # fmt: skip
                t_plan, x_at_plan = world.t_s, world.pieces[0].x_mm
        if plan_ is not None and world.t_s >= plan_.grip_at_t_s:
            break
    assert plan_ is not None and plan_.feasible
    # the true piece position at exactly the grip time (the belt moves at constant speed)
    x_true = x_at_plan + world.belt.speed_mps * 1000.0 * (plan_.grip_at_t_s - t_plan)
    y_true = world.pieces[0].y_mm
    assert abs(plan_.pick_x_mm - x_true) < 2.0 and abs(plan_.pick_y_mm - y_true) < 2.0
