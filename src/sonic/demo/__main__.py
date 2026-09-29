"""Command-line interface: ``python -m sonic.demo playlist``."""

from __future__ import annotations

import argparse
import sys

from .playlist import DEFAULT_DEMO_DIR, build_playlist


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.demo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("playlist", help="one WAV of unseen test clips for live-monitoring demos")
    p.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args(argv)
    manifest = build_playlist(args.per_class)
    for m in manifest:
        print(f"  {m['start_s']:7.1f}-{m['end_s']:7.1f} s  {m['audio_id']}  {m['class_name']}")
    total = manifest[-1]["end_s"] + 3
    print(f"\n{len(manifest)} clips, {total / 60:.1f} min -> {DEFAULT_DEMO_DIR / 'live_test_playlist.wav'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
