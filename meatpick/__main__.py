"""CLI entry point: ``python -m meatpick``."""

from __future__ import annotations

import argparse
import json
import sys

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
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    make_rng(cfg["seed"])  # fail fast on a bad seed

    if args.command == "show-config":
        json.dump(cfg, sys.stdout, indent=2)
        print()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
