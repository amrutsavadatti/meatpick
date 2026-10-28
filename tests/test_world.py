import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.sim import geometry
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World


def make_world(seed=0, overrides=None):
    cfg = load_config(overrides=overrides)
    bank = TextureBank(cfg, make_rng(seed), use_dataset=False)
    return cfg, World(cfg, bank, make_rng(seed + 1000))


def run(world, seconds):
    for _ in range(int(round(seconds / world.cfg["world"]["dt_s"]))):
        world.step()


def test_positions_advance_exactly_by_belt_travel():
    _, w = make_world()
    run(w, 4)
    assert w.pieces
    p = w.pieces[-1]  # newest piece, still far from the belt end
    x0, travel0 = p.x_mm, w.belt.travel_mm
    for i in range(300):
        w.belt.command(0.1 + 0.3 * (i % 7) / 7)  # varying speed
        w.step()
    assert p.x_mm - x0 == pytest.approx(w.belt.travel_mm - travel0, abs=1e-6)


def test_encoder_counts_tenths_of_mm():
    _, w = make_world()
    while w.belt.travel_mm < 1000:
        w.step()
    assert w.belt.encoder == int(w.belt.travel_mm / 0.1 + 1e-6)
    assert 9990 <= w.belt.encoder <= 10100


def test_belt_acceleration_is_limited():
    _, w = make_world()
    w.belt.command(0.0)
    w.step(0.1)
    assert w.belt.speed_mps == pytest.approx(0.25 - 0.5 * 0.1)


def test_belt_speed_is_clamped():
    _, w = make_world()
    w.belt.command(5.0)
    assert w.belt.target_mps == 0.4


def test_60s_run_spawns_50_pieces_plus_minus_10():
    _, w = make_world(seed=0)
    run(w, 60)
    assert abs(w.n_spawned - 50) <= 10


def test_mean_arrival_rate_is_on_target():
    counts = []
    for seed in range(12):
        _, w = make_world(seed=seed)
        run(w, 120)
        counts.append(w.n_spawned / 2)
    assert abs(np.mean(counts) - 50) < 5


def test_clusters_produce_touching_pieces():
    _, w = make_world(seed=3, overrides=["arrivals.burst_prob=1.0", "arrivals.burst_size=[3,3]"])
    run(w, 10)
    gt = w.ground_truth()
    xs = sorted(g.x_mm for g in gt)
    assert len(gt) >= 3
    assert min(np.diff(xs)) < 150  # neighbours sit within a piece length of each other


def test_pick_removes_piece_once():
    _, w = make_world()
    run(w, 6)
    pid = w.pieces[0].id
    assert w.pick(pid) is True
    assert w.pick(pid) is False
    w.step()
    assert all(p.id != pid for p in w.pieces)
    assert w.n_picked == 1


def test_unpicked_piece_is_missed_past_reach_zone_and_removed_at_end():
    _, w = make_world(overrides=["arrivals.rate_per_min=6"])
    run(w, 40)
    assert w.n_missed >= 1
    end_x = w.cfg["world"]["end_x_mm"]
    assert all(p.x_mm - p.half_x_mm < end_x for p in w.pieces)


def test_truth_polygon_matches_piece_geometry_and_fat_side():
    cfg, w = make_world()
    run(w, 30)
    ppm = geometry.px_per_mm(cfg)
    for p in w.pieces:
        gt = next(g for g in w.ground_truth() if g.id == p.id)
        pts = gt.polygon_mm - np.array([gt.x_mm, gt.y_mm])
        diam = max(np.linalg.norm(a - b) for a in pts for b in pts)
        assert 0.93 * gt.length_mm <= diam <= 1.1 * gt.length_mm
        ax = np.array([np.cos(np.radians(gt.angle_deg)), np.sin(np.radians(gt.angle_deg))])
        assert np.ptp(pts @ ax) > 0.9 * np.ptp(pts @ np.array([-ax[1], ax[0]]))  # long along axis
        s = p.sprite
        ys, xs = np.nonzero(s.fat_mask)
        du, dv = xs.mean() - s.cx_px, ys.mean() - s.cy_px
        off = np.array([dv / ppm, -du / ppm])
        normal = np.array([-ax[1], ax[0]])
        assert np.sign(off @ normal) == gt.fat_side


def test_geometry_roundtrip():
    cfg = load_config()
    x, y = 123.4, 321.0
    assert geometry.image_to_belt(cfg, *geometry.belt_to_image(cfg, x, y)) == pytest.approx((x, y))
    assert geometry.image_angle_to_belt(geometry.belt_angle_to_image(37.0)) == pytest.approx(37.0)


def test_same_seed_same_world():
    def snap(seed):
        _, w = make_world(seed=seed)
        run(w, 20)
        return [(p.id, round(p.x_mm, 6), round(p.y_mm, 6), round(p.theta_deg, 6)) for p in w.pieces]

    assert snap(4) == snap(4)
    assert snap(4) != snap(5)
