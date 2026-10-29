import json

import pytest

from meatpick.config import load_config
from meatpick.control.scheduler import read_commands, validate_command_record
from meatpick.runner import run_closed_loop

MINUTES = 0.25


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = load_config()
    metrics = run_closed_loop(cfg, MINUTES, out, detector="classical")
    return cfg, out, metrics


def test_run_writes_commands_events_and_metrics(run):
    _, out, m = run
    assert (out / "metrics.json").exists()
    assert json.loads((out / "metrics.json").read_text())["picked"] == m["picked"]
    assert m["commands"] == len(read_commands(out / "commands.jsonl")) > 0
    assert m["frames"] > 100


def test_every_logged_command_passes_the_robot_interface_schema(run):
    cfg, out, _ = run
    for rec in read_commands(out / "commands.jsonl"):
        assert validate_command_record(rec, cfg) == []


def test_pieces_leave_the_belt_only_through_a_grip_event(run):
    _, out, m = run
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    grips = [e for e in events if e["kind"] == "grip"]
    assert m["picked"] == len(grips)  # the world removed exactly the pieces the arm closed on
    assert len({e["piece_id"] for e in grips}) == len(grips)  # never the same piece twice
    assert m["empty_grips"] == sum(e["kind"] == "empty_grip" for e in events)


def test_picked_plus_missed_never_exceeds_spawned(run):
    _, _, m = run
    assert m["picked"] + m["missed"] <= m["spawned"]
    assert sum(m["miss_causes"].values()) == m["missed"]


def test_a_real_grip_lands_within_the_tolerance(run):
    cfg, _, m = run
    assert m["grip_error_mm_max"] <= cfg["control"]["grip_tolerance_mm"]


def test_the_same_seed_reproduces_the_same_run(run, tmp_path):
    cfg, _, first = run
    again = run_closed_loop(cfg, MINUTES, tmp_path, detector="classical")
    skip = {"wall_s", "realtime_factor", "perception_ms_mean", "perception_ms_p95"}
    assert {k: v for k, v in again.items() if k not in skip} == {
        k: v for k, v in first.items() if k not in skip
    }
