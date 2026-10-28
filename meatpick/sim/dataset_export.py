"""Export synthetic frames as an Ultralytics YOLO segmentation dataset, plus a fixed test set.

Training data: many short *scenes*, each with its own domain randomization (belt colour, lighting,
glare, noise, lens, arrival density, belt speed) and its own seed. Frames inside a scene are spaced
out so they are not near-duplicates, and train/val are split by scene so no scene leaks across.

Test set: not randomized (nominal camera and lighting), consecutive frames in sequences so the
tracker can run on it, with full ground truth saved next to each image.
"""

from __future__ import annotations

import copy
import json
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from meatpick.config import make_rng
from meatpick.sim.camera import Camera, FrameTruth
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World

FRAMES_PER_SCENE = 40
FRAME_STRIDE = 7  # keep every 7th triggered frame (~140 mm of travel) inside a scene
MIN_LABEL_PX = 120  # visible slivers smaller than this are left unlabeled
IMG_SIZE_TRAIN = (640, 360)  # width, height written to the zip; ultralytics trains at 640 anyway
TEST_SEED_BASE = 10_000  # disjoint from training scene seeds
DEV_SEED_BASE = 5_000  # development set: for tuning detectors, never the test set


def randomize_cfg(base: dict, rng: np.random.Generator) -> dict:
    """Return a per-scene config with domain randomization applied."""
    cfg = copy.deepcopy(base)
    u = rng.uniform
    cfg["arrivals"]["rate_per_min"] = float(u(30, 120))
    cfg["arrivals"]["bursty"] = True
    cfg["arrivals"]["burst_prob"] = float(u(0.05, 0.45))
    cfg["arrivals"]["burst_size"] = [2, int(rng.integers(3, 5))]
    cfg["arrivals"]["burst_spacing_mm"] = [float(u(-25, -5)), float(u(10, 40))]
    cfg["belt"]["speed_nominal_mps"] = float(u(0.1, 0.4))
    hsv = cv2.cvtColor(np.uint8([[base["belt"]["color_bgr"]]]), cv2.COLOR_BGR2HSV)[0, 0].astype(
        float
    )
    hsv[0] = (hsv[0] + u(-8, 8)) % 180
    hsv[1] = np.clip(hsv[1] * u(0.7, 1.1), 0, 255)
    hsv[2] = np.clip(hsv[2] * u(0.7, 1.25), 0, 255)
    cfg["belt"]["color_bgr"] = [
        int(v) for v in cv2.cvtColor(hsv.astype(np.uint8)[None, None], cv2.COLOR_HSV2BGR)[0, 0]
    ]
    cam = cfg["camera"]
    cam["randomize"] = True
    cam["vignette"] = float(u(0.1, 0.6))
    cam["light_gradient"] = float(u(0.05, 0.3))
    cam["glare_lobes"] = int(rng.integers(1, 4))
    cam["glare_strength"] = float(u(0.2, 1.4))
    cam["noise_sigma"] = float(u(1.0, 6.0))
    cam["exposure_drift"] = float(u(0.03, 0.12))
    cam["wb_drift"] = float(u(0.01, 0.06))
    cfg["product"]["lean_hue_shift"] = float(u(4, 12))
    cfg["product"]["paleness_range"] = [float(u(-0.15, 0.0)), float(u(0.3, 0.55))]
    return cfg


def test_cfg(base: dict) -> dict:
    """Nominal, non-randomized camera; a denser, burstier line than the demo so touching and
    overlapping pieces are common enough to measure."""
    cfg = copy.deepcopy(base)
    cfg["camera"]["randomize"] = False
    cfg["arrivals"].update(rate_per_min=70, bursty=True, burst_prob=0.2)
    return cfg


def visible_polygon(inst: np.ndarray, piece_id: int) -> np.ndarray | None:
    """Outline of the largest visible component of a piece, in pixels, or None if too small."""
    mask = (inst == piece_id).astype(np.uint8)
    if mask.sum() < MIN_LABEL_PX:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_LABEL_PX * 0.5:
        return None
    cnt = cv2.approxPolyDP(cnt, 1.0, True)[:, 0, :].astype(float)
    return cnt if len(cnt) >= 3 else None


def label_lines(truth: FrameTruth, width: int, height: int) -> list[str]:
    lines = []
    for p in truth.pieces:
        poly = visible_polygon(truth.instance_map, p.id)
        if poly is None:
            continue
        norm = np.clip(poly / [width, height], 0.0, 1.0)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in norm.ravel()))
    return lines


def _write_jpeg(path: Path, img: np.ndarray, quality: int) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _scene_frames(cfg: dict, seed: int, n_frames: int, stride: int, consecutive: bool, keep=None):
    """Yield (frame, truth) from a freshly built scene. ``keep(truth)`` can reject frames."""
    rng = make_rng(seed)
    bank = TextureBank(cfg, rng)
    world = World(cfg, bank, rng)
    cam = Camera(cfg, rng)
    for _ in range(int(rng.uniform(3.0, 6.0) / cfg["world"]["dt_s"])):
        world.step()
    kept = seen = 0
    while kept < n_frames:
        world.step()
        got = cam.maybe_capture(world)
        if got is None:
            continue
        seen += 1
        if consecutive or (seen % stride == 0 and (keep is None or keep(got[1]))):
            kept += 1
            yield got


def export_train_scene(args: tuple) -> int:
    base_cfg, scene_seed, split, out_dir = args
    rng = make_rng(scene_seed + 7919)
    cfg = randomize_cfg(base_cfg, rng)
    out = Path(out_dir)
    n = 0
    for k, (frame, truth) in enumerate(
        _scene_frames(cfg, scene_seed, FRAMES_PER_SCENE, FRAME_STRIDE, False)
    ):
        stem = f"s{scene_seed:04d}_f{k:02d}"
        small = cv2.resize(frame.image, IMG_SIZE_TRAIN, interpolation=cv2.INTER_AREA)
        _write_jpeg(out / "images" / split / f"{stem}.jpg", small, int(rng.integers(78, 96)))
        # Labels come from the full-resolution instance map, normalized to the image size.
        lines = label_lines(truth, frame.image.shape[1], frame.image.shape[0])
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n += 1
    return n


def export_test_sequence(args: tuple) -> list[dict]:
    base_cfg, seq, n_frames, out_dir, *rest = args
    seed_base = rest[0] if rest else TEST_SEED_BASE
    cfg = test_cfg(base_cfg)
    if len(rest) > 1 and rest[1]:
        cfg["arrivals"].update(rest[1])  # e.g. the PRD's nominal 50/min instead of the dense 70/min
    out = Path(out_dir)
    index = []
    for k, (frame, truth) in enumerate(_scene_frames(cfg, seed_base + seq, n_frames, 1, True)):
        stem = f"q{seq:02d}_f{k:03d}"
        _write_jpeg(out / "images" / f"{stem}.jpg", frame.image, 97)
        cv2.imwrite(str(out / "inst" / f"{stem}.png"), (truth.instance_map + 1).astype(np.uint16))
        record = {
            "frame": stem,
            "sequence": seq,
            "index": k,
            "encoder": frame.encoder,
            "t_s": frame.t_s,
            "glare_level": truth.glare_level,
            "pieces": [
                {
                    "id": p.id,
                    "polygon_px": p.polygon_px.round(2).tolist(),
                    "center_px": p.center_px.round(2).tolist(),
                    "x_mm": p.x_mm,
                    "y_mm": p.y_mm,
                    "angle_deg": p.angle_deg,
                    "fat_side": p.fat_side,
                    "visibility": p.visibility,
                    "touching": p.touching,
                    "length_mm": p.length_mm,
                    "width_mm": p.width_mm,
                    "weight_g": p.weight_g,
                }
                for p in truth.pieces
            ],
        }
        (out / "truth" / f"{stem}.json").write_text(json.dumps(record))
        index.append({k_: record[k_] for k_ in ("frame", "sequence", "index", "encoder", "t_s")})
    return index


def export_dataset(
    cfg: dict,
    out_dir: str | Path = "outputs",
    n_train: int = 3400,
    n_val: int = 600,
    n_test: int = 500,
    workers: int = 2,
    base_seed: int = 0,
    progress=None,
) -> dict:
    out_dir = Path(out_dir)
    ds = out_dir / "dataset_yolo"
    test = out_dir / "test_set"
    for d in (ds, test):
        if d.exists():
            shutil.rmtree(d)
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
    for sub in ("images", "inst", "truth"):
        (test / sub).mkdir(parents=True)

    scenes_train = -(-n_train // FRAMES_PER_SCENE)
    scenes_val = -(-n_val // FRAMES_PER_SCENE)
    jobs = [(cfg, base_seed + i, "train", ds) for i in range(scenes_train)]
    jobs += [(cfg, base_seed + scenes_train + i, "val", ds) for i in range(scenes_val)]
    seq_len = 50
    test_jobs = [(cfg, s, seq_len, test) for s in range(-(-n_test // seq_len))]

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n in pool.map(export_train_scene, jobs):
            done += n
            if progress:
                progress(done, len(jobs) * FRAMES_PER_SCENE)
        index = [i for seq in pool.map(export_test_sequence, test_jobs) for i in seq]
    (test / "index.json").write_text(json.dumps(index))

    # trim to the requested counts (scenes round up)
    _trim(ds / "images" / "train", ds / "labels" / "train", n_train)
    _trim(ds / "images" / "val", ds / "labels" / "val", n_val)
    _trim_test(test, n_test)

    (ds / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: meat\n"
    )
    zip_path = out_dir / "dataset_yolo.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # jpegs do not compress further
        for f in sorted(ds.rglob("*")):
            if f.is_file():
                z.write(f, Path("dataset_yolo") / f.relative_to(ds))
    return {
        "train": len(list((ds / "images" / "train").glob("*.jpg"))),
        "val": len(list((ds / "images" / "val").glob("*.jpg"))),
        "test": len(list((test / "images").glob("*.jpg"))),
        "zip": str(zip_path),
    }


def _trim(img_dir: Path, lbl_dir: Path, keep: int) -> None:
    for extra in sorted(img_dir.glob("*.jpg"))[keep:]:
        extra.unlink()
        (lbl_dir / (extra.stem + ".txt")).unlink(missing_ok=True)


def _trim_test(test: Path, keep: int) -> None:
    frames = sorted(test.joinpath("images").glob("*.jpg"))
    for extra in frames[keep:]:
        extra.unlink()
        (test / "inst" / (extra.stem + ".png")).unlink(missing_ok=True)
        (test / "truth" / (extra.stem + ".json")).unlink(missing_ok=True)
    index = json.loads((test / "index.json").read_text())
    kept = {f.stem for f in test.joinpath("images").glob("*.jpg")}
    (test / "index.json").write_text(json.dumps([r for r in index if r["frame"] in kept]))


def export_dev_set(
    cfg: dict,
    out_dir: str | Path = "outputs/dev_set",
    n_seq: int = 8,
    n_frames: int = 25,
    seed_base: int = DEV_SEED_BASE,
    arrivals: dict | None = None,
):
    """Same format as the test set, different seeds. Tune detectors and trackers on this."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images", "inst", "truth"):
        (out / sub).mkdir(parents=True)
    jobs = [(cfg, s, n_frames, out, seed_base, arrivals) for s in range(n_seq)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        index = [i for seq in pool.map(export_test_sequence, jobs) for i in seq]
    (out / "index.json").write_text(json.dumps(index))
    return len(index)
