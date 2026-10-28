"""CLI entry point: ``python -m meatpick``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
