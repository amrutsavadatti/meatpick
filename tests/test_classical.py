import time

import cv2
import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.eval.matching import match_masks
from meatpick.perception.classical import ClassicalDetector
from meatpick.sim.camera import Camera
from meatpick.sim.dataset_export import DEV_SEED_BASE, _scene_frames
from meatpick.sim.dataset_export import (
    test_cfg as dense_line_cfg,
)  # not a test: alias avoids collection
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World
from meatpick.types import Frame

PPM = 1280 / 600


@pytest.fixture(scope="module")
def det():
    return ClassicalDetector(load_config(), PPM)


def _ellipse_blob(shape, centers_axes):
    blob = np.zeros(shape, np.uint8)
    for (cx, cy), (ax, ay) in centers_axes:
        cv2.ellipse(blob, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    return blob


def test_empty_belt_gives_no_detections(det):
    cfg = load_config()
    rng = make_rng(1)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    world.pieces = []
    frame, _ = Camera(cfg, rng).render(world)
    assert det.detect(frame) == []


def test_single_piece_gets_a_tight_mask_and_high_confidence(det):
    cfg = load_config()
    rng = make_rng(2)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    world.pieces = [world._make_piece(170.0, 300.0)]
    frame, truth = Camera(cfg, rng).render(world)
    dets = det.detect(frame)
    assert len(dets) == 1 and dets[0].confidence > 0.8 and not dets[0].touches_border
    gt = truth.instance_map == world.pieces[0].id
    assert (dets[0].mask & gt).sum() / (dets[0].mask | gt).sum() > 0.9
    assert dets[0].mask.shape == frame.image.shape[:2] and dets[0].mask.dtype == bool


@pytest.mark.parametrize("belt_bgr", [[60, 140, 60], [90, 60, 30], [170, 120, 60]])
def test_other_belt_colours_need_no_retuning(belt_bgr):
    cfg = load_config(overrides=[f"belt.color_bgr={belt_bgr}"])
    rng = make_rng(3)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    world.pieces = [world._make_piece(170.0, 300.0)]
    frame, truth = Camera(cfg, rng).render(world)
    dets = ClassicalDetector(cfg, PPM).detect(frame)
    gt = truth.instance_map == world.pieces[0].id
    assert len(dets) == 1 and (dets[0].mask & gt).sum() / (dets[0].mask | gt).sum() > 0.85


def test_piece_cut_by_the_frame_edge_is_flagged(det):
    cfg = load_config()
    rng = make_rng(4)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    world.pieces = [world._make_piece(20.0, 300.0)]  # centre near the top edge of the frame
    frame, _ = Camera(cfg, rng).render(world)
    dets = det.detect(frame)
    assert len(dets) == 1 and dets[0].touches_border


def test_two_overlapping_ellipses_are_split_in_two(det):
    blob = _ellipse_blob((200, 120), [((60, 50), (50, 32)), ((60, 110), (50, 32))])
    parts = det._split(blob, np.zeros(blob.shape + (3,), np.uint8))
    assert len(parts) == 2
    assert all(abs(int(p.sum()) - 5025) / 5025 < 0.2 for p in parts)  # pi * 50 * 32 each
    assert not (parts[0] & parts[1]).any()


def test_a_single_ellipse_is_left_alone(det):
    blob = _ellipse_blob((120, 200), [((100, 60), (80, 40))])
    assert len(det._split(blob, np.zeros(blob.shape + (3,), np.uint8))) == 1


def test_overlap_with_no_waist_is_split_along_its_dark_edge(det):
    # two similar rectangles side by side: the union is a plain rectangle with a dark seam
    img = np.full((140, 200, 3), 170, np.uint8)
    img[:, 99:102] = 70
    blob = np.ones((140, 200), np.uint8)
    parts = det._split(blob, img)
    assert len(parts) == 2


def test_confidence_drops_for_merged_and_ragged_masks(det):
    good = det._confidence(0.97, 9000, False)
    assert good > 0.9
    assert det._confidence(0.97, 40000, False) < 0.1  # far too big for one piece
    assert det._confidence(0.78, 9000, False) < 0.1  # ragged
    assert det._confidence(0.97, 800, False) < 0.3  # too small unless cut by the border
    assert det._confidence(0.97, 800, True) > 0.9


def test_runs_inside_the_time_budget(det):
    cfg = load_config()
    rng = make_rng(5)
    world = World(cfg, TextureBank(cfg, rng, use_dataset=False), rng)
    cam = Camera(cfg, rng)
    frames = []
    while len(frames) < 25:
        world.step()
        got = cam.maybe_capture(world)
        if got:
            frames.append(got[0])
    det.detect(frames[0])
    t0 = time.perf_counter()
    for f in frames:
        det.detect(f)
    assert 1000 * (time.perf_counter() - t0) / len(frames) < 15.0  # PRD: <= 15 ms per frame


def test_fresh_dense_scenes_meet_the_detection_floor(det):
    """End to end on newly rendered scenes (dev seeds, not the saved test set)."""
    cfg = dense_line_cfg(load_config())
    tp = fn = fp = iso_tp = iso_fn = 0
    for seq in range(3):
        for frame, truth in _scene_frames(cfg, DEV_SEED_BASE + 50 + seq, 20, 1, True):
            pieces = [{"id": p.id, "visibility": p.visibility} for p in truth.pieces]
            masks = [d.mask for d in det.detect(Frame(frame.image, frame.encoder, frame.t_s))]
            m = match_masks(masks, truth.instance_map, pieces)
            tp, fn, fp = tp + m.tp, fn + m.fn, fp + m.fp
            for p in truth.pieces:
                if p.visibility >= 0.5 and not p.touching:
                    hit = p.id not in m.missed_ids
                    iso_tp, iso_fn = iso_tp + hit, iso_fn + (not hit)
    assert iso_tp + iso_fn >= 5 and iso_fn == 0  # isolated pieces are always found
    # Overall recall depends on how many hard overlaps a sample holds: a pair that stays in view
    # for 10 frames is a tenth of a 20-frame sequence. Measured 83-94% across samples, so this
    # floor only guards against a broken detector. False positives here are the same merged
    # overlaps counted a second time (a merged mask matches neither piece).
    assert tp / (tp + fn) > 0.75 and fp <= 0.10 * (tp + fp)
