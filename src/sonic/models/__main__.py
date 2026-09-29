"""Command-line interface: ``python -m sonic.models {train <model>,windows,recalibrate,compare}``."""

from __future__ import annotations

import argparse
import sys

from .train import MODEL_MODULES, recalibrate, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.models", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="train, tune, calibrate and save one model")
    p.add_argument("model", choices=sorted(MODEL_MODULES))
    sub.add_parser("windows", help="precompute app-mode windows for validation and test clips")
    r = sub.add_parser("recalibrate", help="redo calibration of a saved model, e.g. svm-20260925-1")
    r.add_argument("version")
    c = sub.add_parser("compare", help="final test-set comparison of the latest version of each model")
    c.add_argument("--models", nargs="+", choices=sorted(MODEL_MODULES))
    args = parser.parse_args(argv)

    if args.command == "windows":
        import numpy as np

        from .data import load_segments
        from .windows import load_windows

        classes = load_segments().classes
        for split in ("validation", "test"):
            w = load_windows(split, classes)
            n = np.diff(w.offsets)
            print(f"{split}: {len(w.audio_ids)} clips, {len(w.X)} windows "
                  f"(median {int(np.median(n))} per clip, max {n.max()}, clips with no audible window: {(n == 0).sum()})")
        return 0
    if args.command == "compare":
        from .compare import run as compare

        compare(args.models)
        return 0
    if args.command == "recalibrate":
        recalibrate(args.version)
        return 0
    run(args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
