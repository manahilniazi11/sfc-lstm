"""Command-line interface: ``python -m sonic.preprocessing {build,check}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..dataset.config import DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR, ConfigError
from .build import DEFAULT_SEGMENTS_DIR, build_segments
from .shortcut_check import main_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.preprocessing", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="preprocess the dataset into training segments")
    p.add_argument("--dataset", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_SEGMENTS_DIR)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_DIR)
    p.add_argument("--workers", type=int, default=4)
    sub.add_parser("check", help="shortcut check: predict the class from recording format only")
    args = parser.parse_args(argv)

    if args.command == "check":
        main_check()
        return 0

    try:
        result = build_segments(args.dataset, args.out, args.metadata, args.workers)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    s = result["settings"]
    print(f"{s.sample_rate} Hz mono, {s.segment_seconds} s segments, noise reduction "
          f"{'on' if s.noise_reduction else 'off'}\n")
    print(f"{'Class':<24}{'train':>7}{'val':>6}{'test':>6}{'total':>7}")
    for name in sorted(result["counts"]):
        c = result["counts"][name]
        print(f"{name:<24}{c['train']:>7}{c['validation']:>6}{c['test']:>6}{sum(c.values()):>7}")
    print(f"\n{result['segments']} segments written to {args.out}")
    if result["problems"]:
        print(f"\n{len(result['problems'])} clips produced no segment:")
        for p in result["problems"][:20]:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
