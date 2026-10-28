from pathlib import Path

import cv2
import numpy as np
import pytest

from meatpick.config import load_config, make_rng
from meatpick.sim.textures import TextureBank

HAS_DATASET = Path("data/raw/meat_detaction/train/images").is_dir()


@pytest.mark.parametrize("kind", ["lean", "fat"])
def test_procedural_fallback_without_dataset(kind):
    bank = TextureBank(load_config(), make_rng(0), use_dataset=False)
    tex = bank.sample(kind, (60, 90), make_rng(1))
    assert tex.shape == (60, 90, 3) and tex.dtype == np.uint8


def test_missing_dataset_dir_falls_back(tmp_path):
    cfg = load_config(overrides=[f"textures.dataset_dir={tmp_path}"])
    bank = TextureBank(cfg, make_rng(0))
    assert not bank.has_real("lean")
    assert bank.sample("lean", 32, make_rng(1)).shape == (32, 32, 3)


def test_fat_is_paler_than_lean_procedural():
    bank = TextureBank(load_config(), make_rng(0), use_dataset=False)
    rng = make_rng(2)
    sat = lambda t: cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[..., 1].mean()  # noqa: E731
    assert sat(bank.sample("fat", 64, rng)) < sat(bank.sample("lean", 64, rng))


def test_bad_kind_rejected():
    bank = TextureBank(load_config(), make_rng(0), use_dataset=False)
    with pytest.raises(ValueError):
        bank.sample("bone", 32, make_rng(0))


def test_deterministic_sampling():
    bank = TextureBank(load_config(), make_rng(0), use_dataset=False)
    a = bank.sample("lean", 48, make_rng(5))
    b = bank.sample("lean", 48, make_rng(5))
    assert np.array_equal(a, b)


@pytest.mark.skipif(not HAS_DATASET, reason="dataset not present")
def test_real_patches_have_no_black_border_and_are_reproducible():
    cfg = load_config()
    bank = TextureBank(cfg, make_rng(0))
    again = TextureBank(cfg, make_rng(0))
    assert bank.has_real("lean") and bank.has_real("fat")
    assert len(bank.patches["lean"]) == len(again.patches["lean"])
    floor = cfg["textures"]["border_black_v"]
    for patch in bank.patches["lean"][:200]:
        assert cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 2].min() > floor // 2
