"""CLI entry point: ``python -m meatpick``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from meatpick.config import load_config, make_rng


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meatpick", description="Meat pick-and-place perception demo.")
    p.add_argument("--config", help="path to a YAML config (default: config/default.yaml)")
    p.add_argument("--seed", type=int, help="override the config seed; reproduces any run")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set robot.vmax_mps=3.0 (repeatable)",
    )
    sub = p.add_subparsers(dest="command")
    sub.add_parser("show-config", help="print the resolved config as JSON")
    tex = sub.add_parser("textures", help="write a 48-sample texture contact sheet")
    tex.add_argument("--out", default="outputs/debug/textures.png")
    tex.add_argument("--procedural", action="store_true", help="ignore the dataset")
    pc = sub.add_parser("pieces", help="write a grid of 50 random pieces")
    pc.add_argument("--out", default="outputs/debug/pieces.png")
    pc.add_argument("--procedural", action="store_true", help="ignore the dataset")
    cs = sub.add_parser("camera-sample", help="render a short camera clip, with and without truth")
    cs.add_argument("--seconds", type=float, default=10.0)
    cs.add_argument("--out", default="outputs/debug/camera_sample.mp4")
    cs.add_argument("--procedural", action="store_true", help="ignore the dataset")
    ds = sub.add_parser("dataset", help="export the YOLO training zip and the fixed test set")
    ds.add_argument("--out", default="outputs")
    ds.add_argument("--train", type=int, default=3400)
    ds.add_argument("--val", type=int, default=600)
    ds.add_argument("--test", type=int, default=500)
    ds.add_argument("--workers", type=int, default=2)
    cb = sub.add_parser("calibrate", help="render calibration boards, calibrate, save the result")
    cb.add_argument("--out", default="outputs/calibration.json")
    cb.add_argument("--debug", default="outputs/debug/calibration.png")
    dv = sub.add_parser("devset", help="export a development set for tuning (not the test set)")
    dv.add_argument("--out", default="outputs/dev_set")
    dv.add_argument("--nominal", action="store_true", help="the PRD's 50/min line, other seeds")
    dd = sub.add_parser("detections-debug", help="contact sheet of detections on a truth set")
    dd.add_argument("--set", default="outputs/dev_set", dest="truth_set")
    dd.add_argument("--out", default="outputs/debug/classical_detections.jpg")
    rn = sub.add_parser("run", help="closed loop: camera, detector, tracker, scheduler, arm")
    rn.add_argument("--minutes", type=float, default=5.0)
    rn.add_argument("--rate", type=float, help="arrivals per minute (default: from config)")
    rn.add_argument("--adaptive", choices=["on", "off"], help="adaptive belt (default: config)")
    rn.add_argument("--detector", choices=["yolo", "classical"])
    rn.add_argument("--out", default="outputs/runs/latest")
    tl = sub.add_parser("truth-loop", help="fixed vs adaptive belt with perfect tracks (fast)")
    tl.add_argument("--minutes", type=float, default=5.0)
    tl.add_argument("--seeds", type=int, nargs="+", default=[200, 201, 202])
    tl.add_argument("--out", default=None)
    td = sub.add_parser("tracks-debug", help="clip of track IDs over one saved sequence")
    td.add_argument("--set", default="outputs/test_set", dest="truth_set")
    td.add_argument("--sequence", type=int, default=0)
    td.add_argument("--detector", default="yolo", choices=["classical", "yolo"])
    td.add_argument("--out", default="outputs/debug/tracks.mp4")
    et = sub.add_parser("eval-tracking", help="ID switches and duplicate tracks")
    et.add_argument("--set", default=None, dest="truth_set", help="saved sequences to replay")
    et.add_argument("--long", type=int, default=0, metavar="N", help="long in-memory run, N pieces")
    et.add_argument("--detector", default="yolo", choices=["classical", "yolo"])
    et.add_argument("--nominal", action="store_true", help="long run at the PRD's 50/min line")
    et.add_argument("--out", default=None)
    pd = sub.add_parser("poses-debug", help="contact sheet of posed detections")
    pd.add_argument("--set", default="outputs/demo_set", dest="truth_set")
    pd.add_argument("--detector", default="yolo", choices=["classical", "yolo"])
    pd.add_argument("--out", default="outputs/debug/poses.jpg")
    ev = sub.add_parser("eval-detector", help="run a detector over a saved truth set")
    ev.add_argument("--set", default="outputs/dev_set", dest="truth_set")
    ev.add_argument("--detector", default="classical", choices=["classical", "yolo"])
    ev.add_argument("--pose", action="store_true", help="also score pose accuracy")
    ev.add_argument("--out", default=None, help="write the report json here")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    rng = make_rng(cfg["seed"])

    if args.command == "show-config":
        json.dump(cfg, sys.stdout, indent=2)
        print()
        return 0
    if args.command == "textures":
        import cv2

        from meatpick.sim.textures import TextureBank, contact_sheet

        bank = TextureBank(cfg, rng, use_dataset=not args.procedural)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, contact_sheet(bank, rng))
        n = {k: len(v) for k, v in bank.patches.items()}
        print(f"wrote {args.out}: {n['lean']} lean / {n['fat']} fat patches")
        return 0
    if args.command == "pieces":
        import cv2

        from meatpick.sim.pieces import generate_piece, piece_grid, rotate
        from meatpick.sim.textures import TextureBank

        bank = TextureBank(cfg, rng, use_dataset=not args.procedural)
        ppm = cfg["camera"]["width_px"] / cfg["camera"]["fov_mm"][0]
        pieces = [
            rotate(generate_piece(cfg, bank, rng, ppm), float(rng.uniform(0, 180)))
            for _ in range(50)
        ]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, piece_grid(pieces))
        print(f"wrote {args.out}")
        return 0
    if args.command == "camera-sample":
        from meatpick.sim.debug import write_camera_clip

        write_camera_clip(cfg, rng, args.out, args.seconds, use_dataset=not args.procedural)
        print(f"wrote {args.out} and its _truth overlay")
        return 0
    if args.command == "dataset":
        from tqdm import tqdm

        from meatpick.sim.dataset_export import export_dataset

        bar = tqdm(total=args.train + args.val, unit="frame")
        last = [0]

        def progress(done: int, total: int) -> None:
            bar.update(done - last[0])
            last[0] = done

        summary = export_dataset(
            cfg, args.out, args.train, args.val, args.test, args.workers, cfg["seed"], progress
        )
        bar.close()
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "calibrate":
        import cv2

        from meatpick.perception.calibration import Calibration
        from meatpick.sim.calibration_target import render_calibration_set
        from meatpick.sim.camera import Camera
        from meatpick.sim.debug import calibration_report

        cam = Camera(cfg, rng)
        tilted, flat, held = render_calibration_set(cam, cfg, rng)
        cal = Calibration.calibrate(tilted, flat)
        cal.save(args.out)
        sheet, summary = calibration_report(cam, cal, tilted, flat, held)
        Path(args.debug).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.debug, sheet)
        print(json.dumps(summary, indent=2))
        print(f"wrote {args.out} and {args.debug}")
        return 0
    if args.command == "devset":
        from meatpick.sim.dataset_export import export_dev_set

        if args.nominal:
            from meatpick.sim.dataset_export import DEV_SEED_BASE

            n = export_dev_set(
                cfg, args.out, seed_base=DEV_SEED_BASE + 100,
                arrivals={"rate_per_min": 50, "burst_prob": 0.1},
            )  # fmt: skip
        else:
            n = export_dev_set(cfg, args.out)
        print(f"wrote {n} frames to {args.out}")
        return 0
    if args.command == "detections-debug":
        import cv2

        from meatpick.eval.matching import frame_stems, load_truth_frame, match_masks
        from meatpick.perception.classical import ClassicalDetector
        from meatpick.types import Frame
        from meatpick.viz.overlay import draw_detections

        det = ClassicalDetector(cfg, cfg["camera"]["width_px"] / cfg["camera"]["fov_mm"][0])
        good, bad = [], []
        for stem in frame_stems(args.truth_set)[::3]:
            tf = load_truth_frame(args.truth_set, stem)
            dets = det.detect(Frame(tf.image, 0, 0))
            res = match_masks([d.mask for d in dets], tf.instance_map, tf.record["pieces"])
            tile = draw_detections(tf.image, dets, [p["polygon_px"] for p in tf.record["pieces"]])
            cv2.putText(tile, f"{stem}  tp {res.tp} fn {res.fn} fp {res.fp}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
            (bad if res.fn or res.fp else good).append(cv2.resize(tile, (640, 360)))
        tiles = (good[:3] + bad[:3] + good[3:6] + bad[3:6])[:6]
        rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles) - 1, 2)]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, np.vstack(rows))
        print(f"wrote {args.out} ({len(good)} clean frames, {len(bad)} with errors sampled)")
        return 0
    if args.command == "run":
        from meatpick.runner import run_closed_loop

        if args.rate:
            cfg["arrivals"]["rate_per_min"] = args.rate
        if args.adaptive:
            cfg["control"]["adaptive_belt"] = args.adaptive == "on"
        result = run_closed_loop(cfg, args.minutes, args.out, args.detector, progress=True)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "truth-loop":
        from meatpick.eval.truth_loop import run_truth_loop

        result = {}
        for label, adaptive in (("fixed", False), ("adaptive", True)):
            runs = [run_truth_loop(cfg, args.minutes, s, adaptive) for s in args.seeds]
            done = sum(r["picked"] + r["missed"] for r in runs)
            result[label] = {
                "seeds": args.seeds,
                "miss_rate": round(sum(r["missed"] for r in runs) / done, 4),
                "timing_misses": sum(r["missed_other"] for r in runs),
                "finger_blocked_misses": sum(r["missed_after_finger_block"] for r in runs),
                "picks_per_min": round(float(np.mean([r["picks_per_min"] for r in runs])), 2),
                "mean_belt_mps": round(float(np.mean([r["mean_belt_mps"] for r in runs])), 3),
                "belt_stopped_fraction": round(
                    float(np.mean([r["belt_stopped_fraction"] for r in runs])), 4
                ),
                "runs": runs,
            }
        print(
            json.dumps(
                {k: {x: y for x, y in v.items() if x != "runs"} for k, v in result.items()},
                indent=2,
            )
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2, default=float))
        return 0
    if args.command == "tracks-debug":
        import cv2

        from meatpick.eval.matching import load_truth_frame
        from meatpick.perception.calibration import Calibration
        from meatpick.perception.factory import Perceiver
        from meatpick.perception.tracker import Tracker
        from meatpick.sim.calibration_target import render_calibration_set
        from meatpick.sim.camera import Camera
        from meatpick.sim.dataset_export import test_cfg
        from meatpick.types import Frame
        from meatpick.viz.overlay import draw_tracks

        ncfg = test_cfg(cfg)
        cam = Camera(ncfg, rng)
        tilted, flat, _ = render_calibration_set(cam, ncfg, rng)
        cal = Calibration.calibrate(tilted, flat)
        perceiver, tracker = Perceiver(cfg, cal, prefer=args.detector), Tracker(cfg, cal)
        index = json.loads((Path(args.truth_set) / "index.json").read_text())
        stems = [r["frame"] for r in index if r["sequence"] == args.sequence]
        size = (cfg["camera"]["width_px"], cfg["camera"]["height_px"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 12.5, size)
        stills = []
        for k, stem in enumerate(stems):
            tf = load_truth_frame(args.truth_set, stem)
            fr = Frame(tf.image, tf.record["encoder"], tf.record["t_s"])
            dets = perceiver.perceive(fr)
            tracks = tracker.update(fr, dets)
            img = draw_tracks(tf.image, tracks, cal, dets)
            writer.write(img)
            if k in (6, 12, 18, 24, 30, 36):
                stills.append(cv2.resize(img, (640, 360)))
        writer.release()
        if len(stills) >= 6:
            sheet = np.vstack([np.hstack(stills[i : i + 2]) for i in range(0, 6, 2)])
            cv2.imwrite(str(Path(args.out).with_suffix(".jpg")), sheet)
        print(f"wrote {args.out} ({len(stems)} frames) using {perceiver.description}")
        return 0
    if args.command == "eval-tracking":
        from meatpick.eval.tracking_eval import evaluate_tracking_long, evaluate_tracking_saved
        from meatpick.perception.calibration import Calibration
        from meatpick.perception.factory import Perceiver
        from meatpick.sim.calibration_target import render_calibration_set
        from meatpick.sim.camera import Camera
        from meatpick.sim.dataset_export import test_cfg
        from meatpick.sim.textures import TextureBank
        from meatpick.sim.world import World

        ncfg = test_cfg(cfg)
        if args.nominal:
            ncfg["arrivals"].update(rate_per_min=50, burst_prob=0.1)
        cam = Camera(ncfg, rng)
        tilted, flat, _ = render_calibration_set(cam, ncfg, rng)
        cal = Calibration.calibrate(tilted, flat)
        perceiver = Perceiver(cfg, cal, prefer=args.detector)
        result = {"detector": perceiver.description}
        if args.truth_set:
            result["saved_set"] = args.truth_set
            result.update(evaluate_tracking_saved(perceiver, args.truth_set, cfg))
        if args.long:
            lrng = make_rng(cfg["seed"] + 777)
            world = World(ncfg, TextureBank(ncfg, lrng), lrng)
            result["long_run"] = evaluate_tracking_long(
                cfg, cal, perceiver, Camera(ncfg, lrng), world, args.long
            )
            result["long_run"]["arrivals"] = dict(ncfg["arrivals"])
        print(json.dumps(result, indent=2, default=float))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2, default=float))
        return 0
    if args.command == "poses-debug":
        import cv2

        from meatpick.eval.matching import frame_stems, load_truth_frame
        from meatpick.perception.calibration import Calibration
        from meatpick.perception.factory import Perceiver
        from meatpick.sim.calibration_target import render_calibration_set
        from meatpick.sim.camera import Camera
        from meatpick.sim.dataset_export import test_cfg
        from meatpick.types import Frame
        from meatpick.viz.overlay import draw_poses

        ncfg = test_cfg(cfg)
        cam = Camera(ncfg, rng)
        tilted, flat, _ = render_calibration_set(cam, ncfg, rng)
        cal = Calibration.calibrate(tilted, flat)
        perceiver = Perceiver(cfg, cal, prefer=args.detector)
        tiles = []
        for stem in frame_stems(args.truth_set)[::7]:
            tf = load_truth_frame(args.truth_set, stem)
            dets = perceiver.perceive(Frame(tf.image, 0, 0))
            if len(dets) >= 2 and len(tiles) < 6:
                tiles.append(cv2.resize(draw_poses(tf.image, dets, cal, cfg), (640, 360)))
        rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles) - 1, 2)]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, np.vstack(rows))
        print(f"wrote {args.out} using {perceiver.description}")
        return 0
    if args.command == "eval-detector":
        from meatpick.eval.detector_eval import evaluate_detector
        from meatpick.perception.calibration import Calibration
        from meatpick.perception.factory import Perceiver
        from meatpick.sim.calibration_target import render_calibration_set
        from meatpick.sim.camera import Camera
        from meatpick.sim.dataset_export import test_cfg

        ncfg = test_cfg(cfg)  # the truth sets use the nominal, non-randomized camera
        cam = Camera(ncfg, rng)
        tilted, flat, _ = render_calibration_set(cam, ncfg, rng)
        cal = Calibration.calibrate(tilted, flat)  # scale and mapping come from calibration
        perceiver = Perceiver(cfg, cal, prefer=args.detector)
        report = evaluate_detector(perceiver.detector, args.truth_set)
        out = {"set": args.truth_set, "detector": perceiver.description,
               "px_per_mm": round(cal.px_per_mm(), 4), **report.summary()}  # fmt: skip
        if args.pose:
            from meatpick.eval.pose_eval import evaluate_pose

            out["pose"] = evaluate_pose(perceiver, args.truth_set)
        print(json.dumps(out, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
