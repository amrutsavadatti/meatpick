import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.eval.pose_eval import angle_error_deg
from meatpick.perception.calibration import Calibration
from meatpick.perception.classical import ClassicalDetector
from meatpick.perception.detector import MaskInstance
from meatpick.perception.factory import Perceiver, build_detector
from meatpick.perception.pose import extract_pose, poses_for, touching_flags
from meatpick.sim import geometry
from meatpick.sim.camera import Camera
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World
from meatpick.types import Detection

PPM = 1280 / 600
HAS_YOLO = Path("models/best.pt").exists()


def _ideal_calibration() -> Calibration:
    """An undistorted, untilted camera: belt x = v / ppm, belt y = 600 - u / ppm."""
    h = np.array([[0, 1 / PPM, 0], [-1 / PPM, 0, 600.0], [0, 0, 1]])
    k = np.array([[1500.0, 0, 640], [0, 1500.0, 360], [0, 0, 1]])
    return Calibration(K=k, dist=np.zeros(5), H=h, image_size=(1280, 720), rms_px=0.0, n_views=0)


def _ellipse_mask(centre_px, length_mm, width_mm, belt_angle_deg):
    mask = np.zeros((720, 1280), np.uint8)
    cv2.ellipse(
        mask,
        (int(centre_px[0]), int(centre_px[1])),
        (round(length_mm * PPM / 2), round(width_mm * PPM / 2)),
        geometry.belt_angle_to_image(belt_angle_deg),
        0, 360, 1, -1,
    )  # fmt: skip
    return mask > 0


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("angle", range(0, 180, 15))
def test_synthetic_masks_recover_their_angle_within_1_degree(cfg, angle):
    cal = _ideal_calibration()
    mask = _ellipse_mask((640, 360), 140.0, 85.0, angle)
    img = np.full((720, 1280, 3), 120, np.uint8)
    det = extract_pose(mask, img, cal, cfg)
    assert angle_error_deg(det.angle_deg, angle) < 1.0
    assert 0.0 <= det.angle_deg < 180.0


def test_synthetic_mask_size_centre_and_weight(cfg):
    cal = _ideal_calibration()
    mask = _ellipse_mask((500, 300), 140.0, 85.0, 40.0)
    det = extract_pose(mask, np.full((720, 1280, 3), 120, np.uint8), cal, cfg)
    cx, cy = 300 / PPM, 600 - 500 / PPM
    assert np.hypot(det.x_mm - cx, det.y_mm - cy) < 0.5
    assert abs(det.length_mm - 140.0) < 1.5 and abs(det.width_mm - 85.0) < 1.5
    expected = np.pi / 4 * 140 * 85 * 22.5 * cfg["product"]["density_g_per_mm3"]
    assert abs(det.est_weight_g - expected) / expected < 0.03


def test_fat_side_follows_the_paler_long_edge(cfg):
    cal = _ideal_calibration()
    angle = 30.0
    mask = _ellipse_mask((640, 360), 150.0, 90.0, angle)
    a = np.radians(angle)
    normal = np.array([-np.sin(a), np.cos(a)])  # belt frame
    ys, xs = np.nonzero(mask)
    bx, by = cal.px_to_belt_mm(xs.astype(float), ys.astype(float))
    s = (bx - bx.mean()) * normal[0] + (by - by.mean()) * normal[1]
    for side in (1, -1):
        img = np.zeros((720, 1280, 3), np.uint8)
        img[:] = (150, 150, 235)  # pink lean in BGR
        fat = side * s > s.max() - 12
        img[ys[fat], xs[fat]] = (205, 228, 242)  # cream fat strip
        assert extract_pose(mask, img, cal, cfg).fat_side == side


def test_touching_flag_uses_the_gripper_clearance():
    a = _ellipse_mask((400, 360), 100.0, 60.0, 0.0)
    for gap_mm, expect in ((10, True), (22, True), (45, False)):
        # same orientation, shifted along belt y so the gap between outlines is gap_mm
        shift_px = round((60 + gap_mm) * PPM)
        b = np.roll(a, shift_px, axis=1)
        assert touching_flags([a, b], PPM, 25.0) == [expect, expect]
    assert touching_flags([a], PPM, 25.0) == [False]


def test_poses_for_flags_touching_and_keeps_confidence(cfg):
    cal = _ideal_calibration()
    a = _ellipse_mask((400, 360), 100.0, 60.0, 0.0)
    b = np.roll(a, round(70 * PPM), axis=1)
    img = np.full((720, 1280, 3), 120, np.uint8)
    dets = poses_for([MaskInstance(a, 0.9), MaskInstance(b, 0.4)], img, cal, cfg)
    assert [d.touching for d in dets] == [True, True]
    assert [round(d.confidence, 2) for d in dets] == [0.9, 0.4]


def _one_piece_frame(cfg, seed):
    rng = make_rng(seed)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    world.pieces = [world._make_piece(170.0, float(rng.uniform(150, 450)))]
    frame, truth = Camera(cfg, rng).render(world)
    return frame, truth.pieces[0]


def test_end_to_end_pose_matches_truth_on_rendered_pieces(cfg):
    from meatpick.sim.calibration_target import render_calibration_set

    rng = make_rng(77)
    cam = Camera(cfg, rng)
    tilted, flat, _ = render_calibration_set(cam, cfg, rng)
    cal = Calibration.calibrate(tilted, flat)
    det = ClassicalDetector(cfg, cal.px_per_mm())
    errs, fat_ok = [], 0
    for seed in range(12):
        rng2 = make_rng(100 + seed)
        world = World(cfg, TextureBank(cfg, rng2, use_dataset=False), rng2)
        world.pieces = [world._make_piece(170.0, float(rng2.uniform(150, 450)))]
        frame, truth = cam.render(world)
        [d] = poses_for(det.detect(frame), frame.image, cal, cfg)
        t = truth.pieces[0]
        errs.append(
            (np.hypot(d.x_mm - t.x_mm, d.y_mm - t.y_mm), angle_error_deg(d.angle_deg, t.angle_deg))
        )
        fat_ok += d.fat_side == t.fat_side
    errs = np.array(errs)
    assert np.median(errs[:, 0]) < 1.0 and errs[:, 0].max() < 3.0  # mm
    assert np.median(errs[:, 1]) < 1.0 and errs[:, 1].max() < 4.0  # degrees
    assert fat_ok >= 11


def test_missing_weights_fall_back_to_classical_with_a_warning(cfg, caplog):
    bad = load_config(overrides=["perception.model_path=models/does_not_exist.pt"])
    with caplog.at_level(logging.WARNING, logger="meatpick.perception"):
        det, desc = build_detector(bad, _ideal_calibration(), prefer="yolo")
    assert isinstance(det, ClassicalDetector) and "fallback" in desc
    assert "not found" in caplog.text


def test_classical_can_be_requested_directly(cfg):
    det, desc = build_detector(cfg, _ideal_calibration(), prefer="classical")
    assert isinstance(det, ClassicalDetector) and desc == "classical"


def _check_detection(d: Detection, shape):
    assert isinstance(d, Detection)
    assert d.mask.shape == shape and d.mask.dtype == bool
    assert 0.0 <= d.angle_deg < 180.0 and d.fat_side in (-1, 1)
    assert d.length_mm >= d.width_mm > 0 and d.est_weight_g > 0 and 0.0 <= d.confidence <= 1.0
    assert isinstance(d.touching, bool)
    assert all(np.isfinite(v) for v in (d.x_mm, d.y_mm, d.length_mm, d.width_mm, d.est_weight_g))


@pytest.mark.skipif(not HAS_YOLO, reason="models/best.pt not present")
def test_both_detectors_return_identical_detection_objects(cfg):
    from meatpick.sim.calibration_target import render_calibration_set

    rng = make_rng(78)
    cam = Camera(cfg, rng)
    tilted, flat, _ = render_calibration_set(cam, cfg, rng)
    cal = Calibration.calibrate(tilted, flat)
    classical, yolo = Perceiver(cfg, cal, "classical"), Perceiver(cfg, cal, "yolo")
    assert "yolo" in yolo.description
    for seed in range(4):
        rng2 = make_rng(200 + seed)
        world = World(cfg, TextureBank(cfg, rng2, use_dataset=False), rng2)
        world.pieces = [world._make_piece(170.0, float(rng2.uniform(150, 450)))]
        frame, _ = cam.render(world)
        a, b = classical.perceive(frame), yolo.perceive(frame)
        assert len(a) == len(b) == 1
        for d in (a[0], b[0]):
            _check_detection(d, frame.image.shape[:2])
        assert type(a[0]) is type(b[0]) and vars(a[0]).keys() == vars(b[0]).keys()
        assert np.hypot(a[0].x_mm - b[0].x_mm, a[0].y_mm - b[0].y_mm) < 3.0
        assert angle_error_deg(a[0].angle_deg, b[0].angle_deg) < 3.0
