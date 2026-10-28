import json
import zipfile

import cv2
import numpy as np
import pytest

from meatpick.config import load_config
from meatpick.sim.dataset_export import export_dataset


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    cfg = load_config()
    summary = export_dataset(cfg, out, n_train=40, n_val=40, n_test=50, workers=2)
    return out, summary


def test_counts_and_zip(exported):
    out, summary = exported
    assert summary["train"] == 40 and summary["val"] == 40 and summary["test"] == 50
    with zipfile.ZipFile(summary["zip"]) as z:
        names = z.namelist()
    assert "dataset_yolo/data.yaml" in names
    assert sum(n.startswith("dataset_yolo/images/train/") for n in names) == 40


def test_every_image_has_a_valid_yolo_seg_label(exported):
    out, _ = exported
    for split in ("train", "val"):
        for img in (out / "dataset_yolo" / "images" / split).glob("*.jpg"):
            assert cv2.imread(str(img)).shape == (360, 640, 3)
            lbl = out / "dataset_yolo" / "labels" / split / (img.stem + ".txt")
            assert lbl.exists()
            for line in lbl.read_text().splitlines():
                parts = line.split()
                assert parts[0] == "0"
                vals = np.array(parts[1:], float)
                assert len(vals) >= 6 and len(vals) % 2 == 0
                assert vals.min() >= 0.0 and vals.max() <= 1.0


def test_few_empty_frames_and_scenes_do_not_leak_across_splits(exported):
    out, _ = exported
    ds = out / "dataset_yolo"
    labels = list((ds / "labels" / "train").glob("*.txt"))
    empty = sum(1 for p in labels if not p.read_text().strip())
    assert empty / len(labels) < 0.2
    scene = lambda split: {p.stem.split("_")[0] for p in (ds / "labels" / split).glob("*.txt")}  # noqa: E731
    assert scene("train").isdisjoint(scene("val"))


def test_test_set_is_sequential_with_full_truth(exported):
    out, _ = exported
    test = out / "test_set"
    index = json.loads((test / "index.json").read_text())
    assert len(index) == 50
    seq0 = [r for r in index if r["sequence"] == 0]
    enc = [r["encoder"] for r in seq0]
    assert np.all(np.diff(enc) == 200)  # exactly 20 mm apart, so the tracker can follow pieces
    recs = [json.loads((test / "truth" / f"{r['frame']}.json").read_text()) for r in seq0]
    with_pieces = next(r for r in recs if r["pieces"])
    assert {"polygon_px", "angle_deg", "fat_side", "visibility", "touching"} <= set(
        with_pieces["pieces"][0]
    )
    seq0_first = seq0[0]
    inst = cv2.imread(str(test / "inst" / f"{seq0_first['frame']}.png"), cv2.IMREAD_UNCHANGED)
    assert inst.dtype == np.uint16 and inst.shape == (720, 1280)
