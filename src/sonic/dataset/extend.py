"""Add new clips to the built dataset without changing any clip that is already in it.

``python -m sonic.dataset build`` draws every class's selection and split
from one random sequence, so new files for one class (e.g. the first
Person Asking for Help clips) would change the selection and split of the
classes after it. That would move clips the GTM model was trained on into
the test set, and change the test set the models were compared on.

``python -m sonic.dataset extend`` therefore only appends:

1. new files from ``raw/custom/`` (not yet in dataset_metadata.csv) for the
   chosen classes are validated like in a build (decodable, long and loud
   enough, not a duplicate, quality not Unusable);
2. their groups are split 70/15/15 per class (a group already in the
   dataset keeps its split, and a group shared by two classes, such as one
   synthetic voice, gets one split for both);
3. they get the next Audio IDs of their class (HLP-00001..., BGN-00301...),
   are copied into the dataset and appended to the metadata; the statistics
   are recomputed.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import report
from ..quality.analyze import assess_file, load_thresholds
from .audio_check import AudioCheckError, probe, rejection_reason
from .builder import (BACKGROUND_CLASS, METADATA_FIELDS, REJECTED_FIELDS, BuildOptions, Selected, _materialize,
                      _rejection, _rel, _write_csv, collect)
from .config import load_classes, load_label_map
from .splitting import SPLITS, assign_splits


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extend(opts: BuildOptions, slugs: list[str]) -> dict[str, Counter]:
    classes = load_classes()
    by_slug = {c.slug: c for c in classes}
    unknown = set(slugs) - set(by_slug)
    if unknown:
        raise ValueError(f"unknown class slug(s) {sorted(unknown)}")
    rows = _read(opts.metadata_dir / "dataset_metadata.csv")
    rejected = _read(opts.metadata_dir / "rejected_files.csv")
    known_paths = {r["source_path"] for r in rows}
    hashes = {r["sha256"]: r["audio_id"] for r in rows}
    split_of = {r["group_id"]: r["split"] for r in rows}  # groups already in the dataset keep their split

    wanted = {by_slug[s].name for s in slugs}
    candidates = [c for c in collect(opts.raw_dir, classes, load_label_map(classes), ["custom"])
                  if c.class_name in wanted and _rel(c.path, opts.raw_dir) not in known_paths]

    quality_cfg = load_thresholds()
    chosen: dict[str, list[Selected]] = defaultdict(list)
    for c in candidates:
        try:
            info = probe(c.path)
        except AudioCheckError as exc:
            rejected.append(_rejection(c, str(exc)))
            continue
        reason = rejection_reason(info, opts.min_duration_s, opts.silence_peak)
        if reason is None and info.sha256 in hashes:
            reason = f"exact duplicate (of {hashes[info.sha256]})"
        if reason is None:
            quality = assess_file(c.path, expect_event=c.class_name != BACKGROUND_CLASS, cfg=quality_cfg)
            if not quality.usable:
                reason = f"quality Unusable: {'; '.join(quality.issues)}"
        if reason:
            rejected.append(_rejection(c, reason))
            continue
        hashes[info.sha256] = _rel(c.path, opts.raw_dir)
        chosen[c.class_name].append(Selected(c, info, quality))

    # The class with the most new clips decides the split of shared groups; stratified per class as in a build.
    rng = random.Random(f"{opts.seed}:extend")
    for class_name in sorted(chosen, key=lambda n: (-len(chosen[n]), n)):
        sizes = Counter(s.candidate.group_id for s in chosen[class_name] if s.candidate.group_id not in split_of)
        split_of.update(assign_splits(dict(sizes), rng, opts.ratios))
        for s in chosen[class_name]:
            s.split = split_of[s.candidate.group_id]

    added: dict[str, Counter] = {}
    order = {s: i for i, s in enumerate(SPLITS)}
    for class_name, selected in chosen.items():
        sound_class = next(c for c in classes if c.name == class_name)
        taken = [int(r["audio_id"].split("-")[1]) for r in rows if r["class_name"] == class_name]
        selected.sort(key=lambda s: (order[s.split], s.candidate.path.as_posix()))
        for n, s in enumerate(selected, start=max(taken, default=0) + 1):
            s.audio_id = f"{sound_class.code}-{n:05d}"
            rows.append(_materialize(s, sound_class.slug, opts))
        added[class_name] = Counter(s.split for s in selected)

    _check_no_leakage(rows)
    for r in rejected:
        r["source_path"] = _rel(Path(r["source_path"]), opts.raw_dir)
    _write_csv(opts.metadata_dir / "dataset_metadata.csv", METADATA_FIELDS, rows)
    _write_csv(opts.metadata_dir / "rejected_files.csv", REJECTED_FIELDS, rejected)
    report.write_statistics(report.compute_statistics(rows, rejected, classes, opts),
                            opts.metadata_dir / "dataset_statistics.json")
    return added


def _check_no_leakage(rows: list[dict]) -> None:
    split_of: dict[str, str] = {}
    for r in rows:
        previous = split_of.setdefault(r["group_id"], r["split"])
        if previous != r["split"]:
            raise AssertionError(f"group {r['group_id']} is in both {previous} and {r['split']}")
