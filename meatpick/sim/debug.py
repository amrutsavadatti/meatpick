"""Debug renders for the simulator (not used by the closed loop)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from meatpick.sim.camera import Camera, FrameTruth
from meatpick.sim.textures import TextureBank
from meatpick.sim.world import World


def draw_truth(img: np.ndarray, truth: FrameTruth) -> np.ndarray:
    out = img.copy()
    for p in truth.pieces:
        color = (0, 255, 255) if p.touching else (0, 255, 0)
        cv2.polylines(out, [p.polygon_px.astype(np.int32)], True, color, 1, cv2.LINE_AA)
        c = tuple(int(v) for v in p.center_px)
        a = np.radians(p.angle_deg)
        # belt-frame axis (cos a, sin a) -> image (u, v) = (-y, x)
        d = np.array([-np.sin(a), np.cos(a)]) * 0.5 * p.length_mm * 2.13
        cv2.line(
            out,
            (int(c[0] - d[0]), int(c[1] - d[1])),
            (int(c[0] + d[0]), int(c[1] + d[1])),
            (0, 0, 255),
            1,
        )
        cv2.putText(
            out,
            f"{p.id} v{p.visibility:.2f}",
            (c[0] - 20, c[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    return out


def write_camera_clip(
    cfg: dict, rng: np.random.Generator, out: str, seconds: float, use_dataset: bool = True
) -> None:
    bank = TextureBank(cfg, rng, use_dataset=use_dataset)
    world = World(cfg, bank, rng)
    cam = Camera(cfg, rng)
    for _ in range(int(4.0 / cfg["world"]["dt_s"])):  # warm up so pieces are already on the belt
        world.step()
    fps = cfg["belt"]["speed_nominal_mps"] * 1000 / cfg["camera"]["trigger_every_mm"]
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (cfg["camera"]["width_px"], cfg["camera"]["height_px"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    clean = cv2.VideoWriter(str(path), fourcc, fps, size)
    truth_w = cv2.VideoWriter(str(path.with_name(path.stem + "_truth.mp4")), fourcc, fps, size)
    still = None
    for _ in range(int(seconds / cfg["world"]["dt_s"])):
        world.step()
        got = cam.maybe_capture(world)
        if got is None:
            continue
        frame, truth = got
        clean.write(frame.image)
        overlay = draw_truth(frame.image, truth)
        truth_w.write(overlay)
        if still is None and len(truth.pieces) >= 3:
            still = overlay
    clean.release()
    truth_w.release()
    if still is not None:
        cv2.imwrite(str(path.with_name(path.stem + "_truth.png")), still)


def calibration_report(cam, cal, tilted, flat, held) -> tuple[np.ndarray, dict]:
    """Contact sheet of calibration views with detected corners, plus summary numbers."""
    from meatpick.perception.calibration import PATTERN, find_corners, flat_board_corners_belt_mm

    errs = []
    for img, origin in held:
        c = find_corners(img)
        x, y = cal.px_to_belt_mm(c[:, 0], c[:, 1])
        t = flat_board_corners_belt_mm(origin)
        errs.append(np.hypot(x - t[:, 0], y - t[:, 1]))
    err = np.concatenate(errs)
    summary = {
        "rms_px": round(cal.rms_px, 3),
        "held_out_mm_mean": round(float(err.mean()), 3),
        "held_out_mm_max": round(float(err.max()), 3),
        "f_px_est": round(float(cal.K[0, 0]), 1),
        "f_px_true": round(float(cam.f_px), 1),
        "k1_est": round(float(cal.dist[0]), 4),
        "k1_true": round(float(cam.dist[0]), 4),
        "n_views": cal.n_views,
    }
    tiles = []
    for img in (tilted[0], tilted[len(tilted) // 2], held[0][0]):
        vis = img.copy()
        c = find_corners(vis)
        cv2.drawChessboardCorners(vis, PATTERN, c.astype(np.float32).reshape(-1, 1, 2), True)
        tiles.append(cv2.resize(vis, (640, 360)))
    text = [
        f"reprojection RMS {summary['rms_px']} px (target < 1)",
        "held-out boards: mean {} mm, max {} mm (target < 1)".format(
            summary["held_out_mm_mean"], summary["held_out_mm_max"]
        ),
        "focal {} est vs {} true | k1 {} vs {}".format(
            summary["f_px_est"], summary["f_px_true"], summary["k1_est"], summary["k1_true"]
        ),
    ]
    info = np.full((360, 640, 3), 30, np.uint8)
    for i, line in enumerate(text):
        cv2.putText(
            info,
            line,
            (12, 60 + 70 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[2], info])])
    return sheet, summary
