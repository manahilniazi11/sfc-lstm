"""Command-line interface: ``python -m sonic.quality {report,file}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import assess_file
from .report import build_report, format_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.quality", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report", help="grade every clip in the built dataset")
    p.add_argument("--workers", type=int, default=4)
    f = sub.add_parser("file", help="grade one audio file")
    f.add_argument("path", type=Path)
    f.add_argument("--background", action="store_true", help="the file is background noise (skip the noise check)")
    args = parser.parse_args(argv)

    if args.command == "file":
        r = assess_file(args.path, expect_event=not args.background)
        print(f"{args.path.name}: {r.grade}")
        for issue in r.issues:
            print(f"  - {issue}")
        for k, v in r.metrics.items():
            print(f"  {k}: {v}")
        return 0
    print(format_summary(build_report(workers=args.workers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
