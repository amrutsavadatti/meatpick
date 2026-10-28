import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.sim.pieces import generate_piece, rotate
from meatpick.sim.textures import TextureBank

PPM = 1280 / 600


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    return cfg, TextureBank(cfg, make_rng(0), use_dataset=False)


def _axis_deg(mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(float)
    pts -= pts.mean(0)
    w, v = np.linalg.eigh(np.cov(pts.T))
    ax = v[:, np.argmax(w)]
    return np.degrees(np.arctan2(ax[1], ax[0])) % 180


def test_mask_matches_alpha_exactly(setup):
    cfg, bank = setup
    rng = make_rng(1)
    for _ in range(10):
        p = rotate(generate_piece(cfg, bank, rng, PPM), rng.uniform(0, 180))
        assert np.array_equal(p.mask, p.sprite[..., 3] >= 128)


def test_dimensions_within_config(setup):
    cfg, bank = setup
    rng = make_rng(2)
    lo, hi = cfg["product"], cfg["product"]
    for _ in range(50):
        p = generate_piece(cfg, bank, rng, PPM)
        assert lo["length_mm"][0] <= p.length_mm <= hi["length_mm"][1]
        assert p.width_mm <= p.length_mm
        assert lo["thickness_mm"][0] <= p.thickness_mm <= hi["thickness_mm"][1]


def test_rotated_axis_and_fat_side_match_ground_truth(setup):
    cfg, bank = setup
    rng = make_rng(3)
    for _ in range(30):
        base = generate_piece(cfg, bank, rng, PPM)
        ang = float(rng.uniform(-180, 360))  # includes half turns
        p = rotate(base, ang)
        diff = abs((_axis_deg(p.mask) - p.angle_deg + 90) % 180 - 90)
        assert diff < 4.0
        a = np.radians(p.angle_deg)
        normal = np.array([-np.sin(a), np.cos(a)])
        ys, xs = np.nonzero(p.fat_mask)
        my, mx = np.nonzero(p.mask)
        offset = np.array([xs.mean() - mx.mean(), ys.mean() - my.mean()])
        assert np.sign(offset @ normal) == p.fat_side


def test_fat_strip_width_in_range(setup):
    cfg, bank = setup
    rng = make_rng(4)
    for _ in range(20):
        p = generate_piece(cfg, bank, rng, PPM)
        # cross-section at the piece center, perpendicular to the long axis (angle 0 -> a column)
        width_mm = p.fat_mask[:, int(round(p.cx_px))].sum() / PPM
        assert 3.0 < width_mm < 23.0
        assert p.width_mm <= 0.75 * p.length_mm + 1e-6


def test_deterministic(setup):
    cfg, bank = setup
    a = generate_piece(cfg, bank, make_rng(9), PPM)
    b = generate_piece(cfg, bank, make_rng(9), PPM)
    assert np.array_equal(a.sprite, b.sprite) and a.fat_side == b.fat_side
