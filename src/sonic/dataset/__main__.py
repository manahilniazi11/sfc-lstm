"""Command-line interface: ``python -m sonic.dataset {download,freesound,recordings,tts,scan,build,extend}``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import download, freesound, report
from .builder import BuildOptions, build, scan
from .config import DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RAW_DIR, ConfigError
from .sources import SOURCES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sonic.dataset", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="show per-source progress")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="download public datasets into SONIC_DATA_DIR/raw")
    p_dl.add_argument("names", nargs="*", help="items to fetch (see --list)")
    p_dl.add_argument("--list", action="store_true", help="show available items and sizes")
    p_dl.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    p_dl.add_argument("--keep-archives", action="store_true", help="keep .zip/.tar.gz after unpacking")
    p_dl.add_argument("--full", action="store_true", help="download whole archives even for [selective] items")
    p_dl.add_argument("--classes", nargs="+", metavar="CLASS",
                      help='FSD50K only: fetch just these classes, e.g. --classes "Panic Scream" Gunshot')

    p_fs = sub.add_parser("freesound", help="top up short classes from Freesound (needs FREESOUND_API_KEY)")
    p_fs.add_argument("--classes", nargs="+", metavar="CLASS", help="default: every class in freesound_queries.json")
    p_fs.add_argument("--max", type=int, help="sounds per class (default: max_sounds in the config)")
    p_fs.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)

    p_rec = sub.add_parser("recordings", help="convert and register the team's own recordings in custom/")
    p_rec.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)

    p_tts = sub.add_parser("tts", help="generate synthetic Person Asking for Help clips (edge-tts) into custom/")
    p_tts.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)

    p_ext = sub.add_parser("extend", help="append new custom/ clips to the built dataset without changing it")
    p_ext.add_argument("classes", nargs="+", metavar="SLUG", help="class slugs, e.g. help_request background_noise")
    p_ext.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    p_ext.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p_ext.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_DIR)

    p_scan = sub.add_parser("scan", help="count candidate files per class without decoding audio")
    _add_source_args(p_scan)

    p_build = sub.add_parser("build", help="build the balanced, split dataset")
    _add_source_args(p_build)
    p_build.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p_build.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_DIR)
    p_build.add_argument("--per-class", type=int, default=300, help="clips per class (0 = no cap)")
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--min-duration", type=float, default=0.3, help="seconds")
    p_build.add_argument("--hardlink", action="store_true", help="hard-link instead of copying files")
    p_build.add_argument("--force", action="store_true", help="replace an existing built dataset")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    try:
        if args.command == "download":
            return _download(args)
        if args.command == "recordings":
            from .recordings import import_recordings

            print(import_recordings(args.raw))
            return 0
        if args.command == "freesound":
            freesound.fetch(args.raw, args.classes, args.max)
            return 0
        if args.command == "tts":
            from .tts import generate

            print(generate(args.raw))
            return 0
        if args.command == "extend":
            from .extend import extend

            added = extend(BuildOptions(raw_dir=args.raw, out_dir=args.out, metadata_dir=args.metadata), args.classes)
            for name, splits in added.items():
                print(f"{name}: +{sum(splits.values())} ({', '.join(f'{k} {v}' for k, v in sorted(splits.items()))})")
            return 0
        if args.command == "scan":
            return _scan(args)
        return _build(args)
    except (ConfigError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--sources", nargs="+", choices=sorted(SOURCES), help="default: all found")


def _download(args) -> int:
    if args.list or not args.names:
        print(download.list_items())
        return 0
    unknown = [n for n in args.names if n not in download.CATALOG]
    if unknown:
        print(f"error: unknown item(s) {unknown}; run with --list", file=sys.stderr)
        return 1
    for name in args.names:
        download.fetch(name, args.raw, args.keep_archives, args.full, args.classes)
    return 0


def _scan(args) -> int:
    counts = scan(BuildOptions(raw_dir=args.raw, sources=args.sources))
    print(f"{'Class':<24}{'files':>7}{'unique':>8}  by source")
    for name, counter in counts.items():
        unique = counter.pop("unique_recordings")
        by_source = ", ".join(f"{s}={n}" for s, n in counter.most_common())
        print(f"{name:<24}{sum(counter.values()):>7}{unique:>8}  {by_source or '-'}")
    return 0


def _build(args) -> int:
    opts = BuildOptions(
        raw_dir=args.raw,
        out_dir=args.out,
        metadata_dir=args.metadata,
        per_class=args.per_class,
        seed=args.seed,
        min_duration_s=args.min_duration,
        sources=args.sources,
        hardlink=args.hardlink,
        force=args.force,
    )
    result = build(opts)
    print(report.format_summary(result.stats))
    print(f"\nDataset written to {opts.out_dir}\nMetadata written to {opts.metadata_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
