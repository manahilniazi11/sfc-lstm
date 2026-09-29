"""Command-line interface: ``python -m sonic.gtm export`` and ``python -m sonic.gtm add <slug>``."""

from __future__ import annotations

import argparse
import sys

from .export import DEFAULT_GTM_DIR, add, export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.gtm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="one playback WAV per class for GTM recording")
    p.add_argument("--seconds", type=int, default=150, help="seconds (= GTM samples) per class")
    a = sub.add_parser("add", help="one more playback file for a trained project; other files stay as they are")
    a.add_argument("slug", help="class slug, e.g. help_request")
    a.add_argument("--seconds", type=int, default=150)
    a.add_argument("--source", help="only segments from this source, e.g. edge-tts")
    args = parser.parse_args(argv)

    if args.command == "add":
        path, n = add(args.slug, args.seconds, args.source)
        print(f"{path}: {n} one-second training segments (manifest.csv updated)")
        return 0
    counts = export(args.seconds)
    for name, n in counts.items():
        print(f"  {name:<24} {n} s  ({n} one-second training segments)")
    print(f"\nWritten to {DEFAULT_GTM_DIR} (play each file while GTM records; see documentation/gtm.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
