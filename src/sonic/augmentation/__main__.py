"""Command-line interface: ``python -m sonic.augmentation build``."""

from __future__ import annotations

import argparse
import sys

from .build import DEFAULT_AUGMENTED_DIR, build_augmented


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.augmentation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="augment training segments (settings: config/augmentation.json)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = build_augmented(workers=args.workers, seed=args.seed)
    print(f"\n{'Class':<24}{'original':>9}{'augmented':>10}{'total':>7}")
    for name in sorted(result["originals"]):
        o, c = result["originals"][name], result["copies"][name]
        print(f"{name:<24}{o:>9}{c:>10}{o + c:>7}")
    print(f"\nWritten to {DEFAULT_AUGMENTED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
