import cv2
import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.sim.camera import Camera
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World


def setup(seed=0, overrides=None):
    cfg = load_config(overrides=overrides)
    rng = make_rng(seed)
    bank = TextureBank(cfg, rng, use_dataset=False)
    world = World(cfg, bank, rng)
    cam = Camera(cfg, rng)
    for _ in range(int(5 / cfg["world"]["dt_s"])):
        world.step()
    return cfg, world, cam


def frames(world, cam, n_steps):
    out = []
    for _ in range(n_steps):
        world.step()
        got = cam.maybe_capture(world)
        if got:
            out.append(got)
    return out


@pytest.fixture(scope="module")
def clip():
    cfg, world, cam = setup(seed=1)
    return cfg, frames(world, cam, 1500)


def test_frames_are_triggered_by_encoder_distance(clip):
    cfg, fs = clip
    enc = [f.encoder for f, _ in fs]
    assert len(enc) > 50
    counts = int(cfg["camera"]["trigger_every_mm"] / cfg["belt"]["mm_per_encoder_count"])
    assert all(counts <= d <= counts + 45 for d in np.diff(enc))  # incl. the first interval
    assert enc[0] % counts < 45  # first frame sits on the trigger grid


def test_frame_shape_and_type(clip):
    cfg, fs = clip
    img = fs[0][0].image
    assert img.shape == (720, 1280, 3) and img.dtype == np.uint8


def test_truth_polygons_match_rendered_pieces_within_2px():
    checked = 0
    for seed in (1, 2, 3):
        cfg, world, cam = setup(seed=seed)
        for _, truth in frames(world, cam, 1500):
            inst = truth.instance_map
            for p in truth.pieces:
                poly = np.zeros(inst.shape, np.uint8)
                cv2.fillPoly(poly, [np.round(p.polygon_px).astype(np.int32)], 1)
                seen = (inst == p.id).astype(np.uint8)
                if p.visibility < 0.99 or poly.sum() == 0:
                    continue
                d_to_poly = cv2.distanceTransform(1 - poly, cv2.DIST_L2, 3)
                d_to_seen = cv2.distanceTransform(1 - seen, cv2.DIST_L2, 3)
                assert d_to_poly[seen > 0].max() <= 2.0
                assert d_to_seen[poly > 0].max() <= 2.0
                checked += 1
    assert checked >= 15


def test_truth_contains_only_world_pieces_and_valid_fields(clip):
    _, fs = clip
    for _, truth in fs:
        for p in truth.pieces:
            assert 0.0 < p.visibility <= 1.0
            assert p.fat_side in (-1, 1) and 0 <= p.angle_deg < 180


def test_same_seed_same_frame():
    a = frames(*setup(seed=5)[1:], 300)
    b = frames(*setup(seed=5)[1:], 300)
    assert len(a) == len(b) and np.array_equal(a[-1][0].image, b[-1][0].image)


def test_noise_and_drift_off_gives_identical_still_frames():
    off = {f"camera.effects.{k}": False for k in ("sensor_noise", "exposure_drift")}
    cfg, world, cam = setup(seed=2, overrides=[f"{k}={str(v).lower()}" for k, v in off.items()])
    a, _ = cam.render(world)
    b, _ = cam.render(world)
    assert np.array_equal(a.image, b.image)


def test_sensor_noise_on_changes_still_frames():
    cfg, world, cam = setup(seed=2)
    a, _ = cam.render(world)
    b, _ = cam.render(world)
    assert not np.array_equal(a.image, b.image)


def test_motion_blur_softens_travel_direction():
    def energy(blur):
        cfg, world, cam = setup(
            seed=3,
            overrides=[
                f"camera.effects.motion_blur={str(blur).lower()}",
                "camera.effects.sensor_noise=false",
            ],
        )
        world.belt.speed_mps = world.belt.target_mps = 0.4
        img = cam.render(world)[0].image.astype(np.float32)
        return np.abs(np.diff(img, axis=0)).mean()

    assert energy(True) < energy(False)


def test_lens_model_round_trip():
    cfg, world, cam = setup(seed=4)
    pts = np.array([[100.0, 80.0], [640.0, 360.0], [1200.0, 650.0]])
    final = cam.project(pts)
    # sample the remap maps at the projected pixel: they must point back at the ideal pixel
    for (u, v), (iu, iv) in zip(final, pts, strict=True):
        mu = cv2.getRectSubPix(cam.map_x, (1, 1), (float(u), float(v)))[0, 0] - cam.margin
        mv = cv2.getRectSubPix(cam.map_y, (1, 1), (float(u), float(v)))[0, 0] - cam.margin
        assert abs(mu - iu) < 0.5 and abs(mv - iv) < 0.5


def test_cut_off_piece_casts_no_wrapped_shadow():
    """A piece clipped by the canvas edge must not leave a shadow smear on its far side."""
    cfg, world, cam = setup(seed=6)
    piece = world.pieces[0]
    world.pieces = [piece]
    # centre 40 px above the bottom canvas edge so the piece is cut off by it
    piece.x_mm = (cam.H + cam.margin - 40) / cam.ppm
    base = cam._background(world.belt.travel_mm)
    clean = base.copy()
    inst, _ = cam._composite(base, world)
    changed_rows = np.nonzero((np.abs(base - clean).max(axis=2) > 0.5).any(axis=1))[0]
    piece_rows = np.nonzero((inst == piece.id).any(axis=1))[0]
    assert piece_rows.size and changed_rows.size
    assert changed_rows.min() >= piece_rows.min() - 18  # only the soft shadow edge may sit above
