import numpy as np
import pytest

from meatpick.config import load_config
from meatpick.control.arm_model import ArmState
from meatpick.control.belt_controller import BeltController
from meatpick.types import Track


class FakeArm:
    """The controller only looks at these three things."""

    def __init__(self, free_at_s=0.0, state=ArmState.IDLE):
        self.free_at_s, self.state = free_at_s, state

    @property
    def busy(self):
        return self.state != ArmState.IDLE


def piece(i, x, status="confirmed"):
    return Track(i, x, 300.0, 1000, 0.0, 1, 1.0, 99.0, status)


def ctl(**over):
    return BeltController(load_config(overrides=[f"control.{k}={v}" for k, v in over.items()]))


def step(c, tracks, arm=None, now=0.0, speed=0.25, skip=frozenset(), dt=0.01):
    return c.update(now, dt, 1000, tracks, set(skip), arm or FakeArm(), speed)


def test_a_disabled_controller_just_holds_the_nominal_speed():
    c = ctl(adaptive_belt="false")
    assert step(c, [piece(1, 990)], FakeArm(free_at_s=5.0)) == pytest.approx(0.25)


def test_with_nothing_waiting_the_belt_runs_at_its_nominal_speed():
    assert step(ctl(), []) == pytest.approx(0.25)
    assert step(ctl(), [piece(1, 500, "tentative")]) == pytest.approx(0.25)


def test_with_plenty_of_slack_the_belt_is_not_sped_up_beyond_the_cap():
    # piece far upstream, arm idle: slack is huge, but the controller never exceeds its cap
    assert step(ctl(), [piece(1, 300)]) == pytest.approx(0.25)
    assert step(ctl(belt_speed_max_mps=0.32), [piece(1, 300)]) <= 0.32 + 1e-9


def test_little_slack_slows_the_belt_in_proportion_to_the_shortfall():
    arm = FakeArm(free_at_s=1.0, state=ArmState.MOVING_TO_BIN)
    mild = step(ctl(), [piece(1, 700)], arm)  # slack = 1.2 - 1.0 = 0.2 s: below the 1.6 s setpoint
    severe = step(ctl(), [piece(1, 850)], arm)  # slack = 0.6 - 1.0 < 0
    assert 0.0 <= severe <= mild < 0.25


def test_command_always_stays_inside_the_belt_limits():
    c = ctl()
    for x in (100, 400, 700, 900, 999):
        for free in (0.0, 0.5, 2.0, 5.0):
            v = step(c, [piece(1, x)], FakeArm(free_at_s=free, state=ArmState.MOVING_TO_BIN))
            assert 0.0 <= v <= 0.4 + 1e-9


def test_an_imminent_miss_stops_the_belt():
    arm = FakeArm(free_at_s=1.2, state=ArmState.MOVING_TO_PICK)
    c = ctl()
    assert step(c, [piece(1, 850)], arm) == 0.0 and c.stopped and c.stop_count == 1


def test_the_stop_is_held_until_the_arm_is_free_then_until_it_grips():
    c = ctl()
    arm = FakeArm(free_at_s=1.2, state=ArmState.MOVING_TO_PICK)
    tracks = [piece(1, 850)]
    step(c, tracks, arm)
    assert c.stopped
    arm.state = ArmState.MOVING_TO_BIN  # still finishing the previous cycle
    assert step(c, tracks, arm, speed=0.0) == 0.0 and c._stop_phase == 1
    arm.state = ArmState.IDLE  # arm free: the scheduler gets its chance to dispatch
    assert step(c, tracks, arm, speed=0.0) == 0.0 and c._stop_phase == 2
    arm.state = ArmState.MOVING_TO_PICK  # dispatched: keep the piece still while the arm goes
    assert step(c, tracks, arm, speed=0.0) == 0.0 and c.stopped
    arm.state = ArmState.GRIPPING  # gripped: let the belt go
    step(c, tracks, arm, speed=0.0)
    assert not c.stopped


def test_the_stop_is_dropped_if_the_arm_is_free_and_nothing_gets_dispatched():
    c = ctl()
    arm = FakeArm(free_at_s=1.2, state=ArmState.MOVING_TO_PICK)
    step(c, [piece(1, 850)], arm)
    arm.state = ArmState.IDLE
    for _ in range(40):  # 0.4 s with the arm idle and no pick
        step(c, [piece(1, 850)], arm, speed=0.0)
    assert not c.stopped


def test_a_piece_the_scheduler_will_not_pick_cannot_hold_the_belt():
    """Regression: a finger-blocked piece used to stop the belt forever."""
    arm = FakeArm(free_at_s=1.2, state=ArmState.MOVING_TO_PICK)
    c = ctl()
    assert step(c, [piece(1, 850)], arm, skip={1}) == pytest.approx(0.25) and not c.stopped


def test_integral_does_not_wind_up_while_the_output_is_saturated():
    c = ctl()
    arm = FakeArm(free_at_s=3.0, state=ArmState.MOVING_TO_BIN)
    for _ in range(500):  # 5 s of a shortfall that pins the output at 0
        step(c, [piece(1, 400)], arm)
    assert abs(c.integral) <= c.i_max + 1e-9 and c.integral > -c.i_max * 1.01


# --- the speed profile handed to the planner -----------------------------------------------
def numeric_travel(v0, v1, accel, dt, n=20000):
    ts = np.linspace(0, dt, n)
    v = np.where(v1 >= v0, np.minimum(v0 + accel * ts, v1), np.maximum(v0 - accel * ts, v1))
    return float(np.trapezoid(v, ts))


@pytest.mark.parametrize(
    "v0,v1", [(0.25, 0.25), (0.25, 0.0), (0.0, 0.25), (0.25, 0.1), (0.1, 0.32)]
)
@pytest.mark.parametrize("dt", [0.1, 0.5, 1.5])
def test_predicted_travel_matches_integrating_the_speed_ramp(v0, v1, dt):
    c = ctl()
    c.speed_mps, c.command_mps = v0, v1
    expect = numeric_travel(v0 * 1000, v1 * 1000, c.accel * 1000, dt)
    assert c.predict_travel(dt) == pytest.approx(expect, rel=1e-3, abs=0.05)


def test_predicted_speed_never_changes_faster_than_the_acceleration_limit():
    c = ctl()
    c.speed_mps, c.command_mps = 0.25, 0.0
    dts = np.linspace(0.0, 1.0, 101)
    d = np.array([c.predict_travel(t) for t in dts])
    speed = np.diff(d) / np.diff(dts) / 1000.0  # m/s
    assert np.max(np.abs(np.diff(speed)) / np.diff(dts)[1:]) <= c.accel * 1.05


# --- the closed loop with perfect tracks, on seeds NOT used for tuning ---------------------
def test_adaptive_belt_cuts_timing_misses_to_under_a_fifth_and_never_deadlocks():
    from meatpick.eval.truth_loop import run_truth_loop

    cfg = load_config()
    fixed = [run_truth_loop(cfg, 5.0, seed=s, adaptive=False) for s in (200, 201, 202)]
    adapt = [run_truth_loop(cfg, 5.0, seed=s, adaptive=True) for s in (200, 201, 202)]
    f_time = sum(r["missed_other"] for r in fixed)
    a_time = sum(r["missed_other"] for r in adapt)
    assert f_time >= 40 and a_time < 0.2 * f_time  # the timing misses the belt can fix
    # the whole miss rate improves and so does throughput (finger-blocked pieces remain a floor)
    f_miss = sum(r["missed"] for r in fixed) / sum(r["picked"] + r["missed"] for r in fixed)
    a_miss = sum(r["missed"] for r in adapt) / sum(r["picked"] + r["missed"] for r in adapt)
    assert a_miss < f_miss
    assert np.mean([r["picks_per_min"] for r in adapt]) > np.mean(
        [r["picks_per_min"] for r in fixed]
    )
    assert max(r["belt_stopped_fraction"] for r in adapt) < 0.25  # the belt keeps moving
