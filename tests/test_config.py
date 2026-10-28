import json
import subprocess
import sys

import pytest

from meatpick.config import DEFAULT_CONFIG, load_config, make_rng


def test_default_config_loads():
    cfg = load_config()
    assert cfg["belt"]["speed_nominal_mps"] == 0.25
    assert cfg["robot"]["vmax_mps"] == 2.0


def test_override_changes_value():
    cfg = load_config(overrides=["robot.vmax_mps=3.5", "control.adaptive_belt=false"])
    assert cfg["robot"]["vmax_mps"] == 3.5
    assert cfg["control"]["adaptive_belt"] is False


def test_unknown_override_rejected():
    with pytest.raises(KeyError):
        load_config(overrides=["robot.nope=1"])


def test_same_seed_same_numbers():
    assert make_rng(7).random() == make_rng(7).random()


def test_cli_help_runs():
    cmd = [sys.executable, "-m", "meatpick", "--help"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0
    assert "meatpick" in out.stdout


def test_cli_yaml_change_changes_behavior(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    text = DEFAULT_CONFIG.read_text().replace("vmax_mps: 2.0", "vmax_mps: 4.0")
    cfg_file.write_text(text)
    out = subprocess.run(
        [sys.executable, "-m", "meatpick", "--config", str(cfg_file), "show-config"],
        capture_output=True,
        text=True,
    )
    assert json.loads(out.stdout)["robot"]["vmax_mps"] == 4.0
