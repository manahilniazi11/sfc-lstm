"""Command-line interface: ``python -m sonic.features {build,list}``."""

from __future__ import annotations

import argparse
import sys

from .build import build_features
from .extract import feature_names
from .settings import load_feature_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.features", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="extract features for all segments")
    p.add_argument("--workers", type=int, default=4)
    sub.add_parser("list", help="print the feature names")
    args = parser.parse_args(argv)

    if args.command == "list":
        print("\n".join(feature_names(load_feature_settings())))
        return 0
    r = build_features(workers=args.workers)
    print(f"{r['rows']} segments -> {r['features']} summary features + log-mel {r['mel_shape']}")
    print(f"non-finite values: {r['non_finite']}")
    print(f"written to {r['out_dir']}")
    return 0 if r["non_finite"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
