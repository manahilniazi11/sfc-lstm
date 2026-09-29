"""Command-line interface: ``python -m sonic.comparison prepare | serve | report``."""

from __future__ import annotations

import argparse
import sys

from . import runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.comparison", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="copy the test clips and the GTM page into the run folder")
    p.add_argument("--per-class", type=int, default=None, help="only the first N clips of each class (default: all)")
    p = sub.add_parser("serve", help="serve the GTM page and save its results")
    p.add_argument("--port", type=int, default=8003)
    p = sub.add_parser("report", help="run the Python model and write reports/model_comparison.md and .csv")
    p.add_argument("--per-class", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        clips = runner.prepare(args.per_class)
        print(f"{len(clips)} test clips in {runner.RUN_DIR}. Next: python -m sonic.comparison serve")
    elif args.command == "serve":
        runner.serve(args.port)
    else:
        from .report import REPORT_CSV, REPORT_MD, run  # imports TensorFlow

        s = run(args.per_class)
        print(f"Python {s['python']['accuracy']:.1%} / GTM {s['gtm']['accuracy']:.1%} / final "
              f"{s['final']['accuracy']:.1%} on {s['n']} recordings; {len(s['major'])} major disagreements")
        print(f"Written {REPORT_MD} and {REPORT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
