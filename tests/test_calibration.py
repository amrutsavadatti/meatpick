import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.perception.calibration import (
    Calibration,
    CalibrationError,
    find_corners,
    flat_board_corners_belt_mm,
)
from meatpick.sim.calibration_target import render_calibration_set
from meatpick.sim.camera import Camera


@pytest.fixture(scope="module", params=[11, 12])
def session(request):
    cfg = load_config()
    rng = make_rng(request.param)
    cam = Camera(cfg, rng)  # lens, tilt and lighting differ per seed
    tilted, flat, held = render_calibration_set(cam, cfg, rng)
    return cam, Calibration.calibrate(tilted, flat), tilted, flat, held


def test_reprojection_error_under_1px(session):
    _, cal, *_ = session
    assert cal.rms_px < 1.0


def test_mapped_corners_within_1mm_on_boards_the_fit_never_saw(session):
    _, cal, _, _, held = session
    worst = 0.0
    for img, origin in held:
        c = find_corners(img)
        assert c is not None
        x, y = cal.px_to_belt_mm(c[:, 0], c[:, 1])
        truth = flat_board_corners_belt_mm(origin)
        worst = max(worst, float(np.hypot(x - truth[:, 0], y - truth[:, 1]).max()))
    assert worst < 1.0


def test_detected_corners_sit_where_the_sim_camera_puts_them(session):
    cam, _, _, _, held = session
    for img, origin in held:
        truth = flat_board_corners_belt_mm(origin)
        expected = cam.belt_to_px(truth[:, 0], truth[:, 1])
        assert np.abs(find_corners(img) - expected).max() < 0.5  # px


def test_whole_field_of_view_matches_the_true_camera(session):
    cam, cal, *_ = session
    gx, gy = np.meshgrid(np.linspace(5, 330, 15), np.linspace(10, 590, 25))
    x, y = gx.ravel(), gy.ravel()
    px = cam.belt_to_px(x, y)
    ok = (px[:, 0] > 0) & (px[:, 0] < 1280) & (px[:, 1] > 0) & (px[:, 1] < 720)
    bx, by = cal.px_to_belt_mm(px[ok, 0], px[ok, 1])
    assert np.hypot(bx - x[ok], by - y[ok]).max() < 0.5  # mm


def test_forward_and_inverse_are_consistent(session):
    _, cal, *_ = session
    x, y = np.array([20.0, 150.0, 300.0]), np.array([50.0, 300.0, 550.0])
    u, v = cal.belt_mm_to_px(x, y)
    bx, by = cal.px_to_belt_mm(u, v)
    assert np.allclose(bx, x, atol=1e-4) and np.allclose(by, y, atol=1e-4)
    assert isinstance(cal.px_to_belt_mm(640.0, 360.0)[0], float)  # scalars stay scalars


def test_focal_length_and_distortion_are_recovered(session):
    cam, cal, *_ = session
    assert abs(cal.K[0, 0] / cam.f_px - 1) < 0.01
    assert abs(cal.dist[0] - cam.dist[0]) < 0.02
    assert np.allclose(cal.K[:2, 2], [cam.W / 2, cam.H / 2], atol=2.0)


def test_save_and_load_round_trip(session, tmp_path):
    _, cal, *_ = session
    cal.save(tmp_path / "c.json")
    again = Calibration.load(tmp_path / "c.json")
    assert np.allclose(again.px_to_belt_mm(500.0, 300.0), cal.px_to_belt_mm(500.0, 300.0))


def test_no_board_in_the_images_raises():
    blank = np.full((720, 1280, 3), 120, np.uint8)
    flat = [(blank, (15.0, 285.0))]
    with pytest.raises(CalibrationError):
        Calibration.calibrate([blank] * 8, flat)


def test_still_accurate_with_heavy_noise_and_vignetting():
    cfg = load_config(overrides=["camera.noise_sigma=9.0", "camera.vignette=0.6"])
    rng = make_rng(21)
    cam = Camera(cfg, rng)
    tilted, flat, held = render_calibration_set(cam, cfg, rng)
    cal = Calibration.calibrate(tilted, flat)
    c = find_corners(held[0][0])
    x, y = cal.px_to_belt_mm(c[:, 0], c[:, 1])
    truth = flat_board_corners_belt_mm(held[0][1])
    assert cal.rms_px < 1.0 and np.hypot(x - truth[:, 0], y - truth[:, 1]).max() < 1.0


def test_view_span_matches_where_the_tilted_camera_really_looks(session):
    cam, cal, *_ = session
    x_top, x_bottom = cal.view_span_x_mm()
    u, _ = cal.belt_mm_to_px(x_top, 300.0)
    _, v_top = cam.belt_to_px(np.array([x_top]), np.array([cal.px_to_belt_mm(640.0, 0.0)[1]]))[0]
    _, v_bottom = cam.belt_to_px(
        np.array([x_bottom]), np.array([cal.px_to_belt_mm(640.0, 719.0)[1]])
    )[0]
    assert abs(v_top - 0.0) < 1.0 and abs(v_bottom - 719.0) < 1.0
    assert x_bottom - x_top > 300  # a real span, in belt mm
