"""The dataset build pipeline.

collect -> drop label conflicts -> select & validate per class -> split by
group -> assign Audio IDs -> copy files -> write metadata and statistics.
"""

from __future__ import annotations

import csv
import logging
import os
import random
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from . import report
from ..quality.analyze import QualityResult, assess_file, load_thresholds
from .audio_check import AudioCheckError, AudioInfo, probe, rejection_reason
from .candidates import Candidate
from .config import (
    DEFAULT_METADATA_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_RAW_DIR,
    SoundClass,
    load_classes,
    load_label_map,
)
from .sources import SOURCES
from .splitting import DEFAULT_RATIOS, SPLITS, assign_splits

log = logging.getLogger(__name__)

BACKGROUND_CLASS = "Background Noise"  # noise by definition: skip the excessive-noise check

METADATA_FIELDS = [
    "audio_id", "filename", "class_name", "class_slug", "split",
    "source_dataset", "source_label", "source_path", "original_filename",
    "origin_id", "group_id", "freesound_id", "license",
    "format", "duration_s", "sample_rate", "channels", "bit_depth",
    "file_size_bytes", "sha256", "peak_amplitude", "quality_grade", "quality_issues",
    "environment", "device", "distance_m", "speaker",
    "is_augmented", "parent_audio_id", "notes",
]
REJECTED_FIELDS = ["source_dataset", "source_path", "class_name", "group_id", "reason"]


@dataclass
class BuildOptions:
    raw_dir: Path = DEFAULT_RAW_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    metadata_dir: Path = DEFAULT_METADATA_DIR
    per_class: int = 300  # 0 means "take everything"
    seed: int = 42
    ratios: tuple[float, float, float] = DEFAULT_RATIOS
    min_duration_s: float = 0.3
    silence_peak: float = 0.001  # about -60 dBFS
    sources: list[str] | None = None  # None means all
    hardlink: bool = False
    force: bool = False


@dataclass
class Selected:
    candidate: Candidate
    info: AudioInfo
    quality: QualityResult
    split: str = ""
    audio_id: str = ""


@dataclass
class BuildResult:
    rows: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def collect(
    raw_dir: Path,
    classes: list[SoundClass],
    label_map: dict[str, dict[str, str | None]],
    sources: list[str] | None = None,
) -> list[Candidate]:
    """Run every source adapter and return candidates in a stable order."""
    names = sources or list(SOURCES)
    unknown = set(names) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown source(s) {sorted(unknown)}; choose from {sorted(SOURCES)}")

    candidates: list[Candidate] = []
    for name in names:
        root = raw_dir / name
        if not root.is_dir():
            log.info("%s: not found at %s, skipped", name, root)
            continue
        if name == "custom":
            mapping = {c.slug: c.name for c in classes} | {c.name.lower(): c.name for c in classes}
        else:
            mapping = label_map.get(name, {})
        found = list(SOURCES[name](root, mapping))
        log.info("%s: %d candidate files", name, len(found))
        candidates.extend(found)
    candidates.sort(key=lambda c: (c.source, c.path.as_posix()))
    return candidates


def drop_conflicting_groups(candidates: list[Candidate]) -> tuple[list[Candidate], list[dict]]:
    """Remove recordings that different datasets label as different classes."""
    classes_per_group: dict[str, set[str]] = defaultdict(set)
    for c in candidates:
        classes_per_group[c.group_id].add(c.class_name)
    conflicts = {g for g, names in classes_per_group.items() if len(names) > 1}
    kept, rejected = [], []
    for c in candidates:
        if c.group_id in conflicts:
            labels = " vs ".join(sorted(classes_per_group[c.group_id]))
            rejected.append(_rejection(c, f"label conflict across sources ({labels})"))
        else:
            kept.append(c)
    return kept, rejected


def select_for_class(
    candidates: list[Candidate],
    target: int,
    rng: random.Random,
    validate,
) -> list[Selected]:
    """Pick up to ``target`` valid clips, one group at a time.

    Round-robin over groups: every original recording contributes one clip
    before any recording contributes a second, which maximises diversity.
    Groups with lower ``priority`` (the team's own recordings) go first.
    Files are validated lazily, so huge sources are not fully decoded.
    ``validate`` returns ``(AudioInfo, QualityResult)`` or None to reject.
    """
    by_group: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_group[c.group_id].append(c)
    groups = sorted(by_group)
    rng.shuffle(groups)
    groups.sort(key=lambda g: min(c.priority for c in by_group[g]))
    queues = []
    for g in groups:
        clips = by_group[g]
        rng.shuffle(clips)
        queues.append(deque(clips))

    limit = target or float("inf")
    selected: list[Selected] = []
    while queues and len(selected) < limit:
        still_open = []
        for queue in queues:
            if len(selected) >= limit:
                break
            while queue:
                candidate = queue.popleft()
                checked = validate(candidate)
                if checked is not None:
                    selected.append(Selected(candidate, *checked))
                    break
            if queue:
                still_open.append(queue)
        queues = still_open
    return selected


def build(opts: BuildOptions) -> BuildResult:
    classes = load_classes()
    label_map = load_label_map(classes)
    rng = random.Random(opts.seed)
    _prepare_output(opts)

    candidates = collect(opts.raw_dir, classes, label_map, opts.sources)
    candidates, rejected = drop_conflicting_groups(candidates)

    seen_hashes: dict[str, str] = {}
    quality_cfg = load_thresholds()

    def validate(c: Candidate) -> tuple[AudioInfo, QualityResult] | None:
        try:
            info = probe(c.path)
        except AudioCheckError as exc:
            rejected.append(_rejection(c, str(exc)))
            return None
        reason = rejection_reason(info, opts.min_duration_s, opts.silence_peak)
        if reason is None and info.sha256 in seen_hashes:
            reason = f"exact duplicate (of {seen_hashes[info.sha256]})"
        if reason:
            rejected.append(_rejection(c, reason))
            return None
        # SRS Step 13 quality grade; Unusable clips never enter the dataset.
        quality = assess_file(c.path, expect_event=c.class_name != BACKGROUND_CLASS, cfg=quality_cfg)
        if not quality.usable:
            rejected.append(_rejection(c, f"quality Unusable: {'; '.join(quality.issues)}"))
            return None
        seen_hashes[info.sha256] = _rel(c.path, opts.raw_dir)
        return info, quality

    by_class: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_class[c.class_name].append(c)

    selected_by_class: dict[str, list[Selected]] = {}
    for sound_class in classes:
        chosen = select_for_class(by_class.get(sound_class.name, []), opts.per_class, rng, validate)
        group_sizes = Counter(s.candidate.group_id for s in chosen)
        assignment = assign_splits(dict(group_sizes), rng, opts.ratios)
        for s in chosen:
            s.split = assignment[s.candidate.group_id]
        _assign_audio_ids(chosen, sound_class)
        selected_by_class[sound_class.name] = chosen

    _check_no_leakage(s for chosen in selected_by_class.values() for s in chosen)

    rows = []
    slugs = {c.name: c.slug for c in classes}
    for class_name, chosen in selected_by_class.items():
        for s in chosen:
            rows.append(_materialize(s, slugs[class_name], opts))

    for r in rejected:
        r["source_path"] = _rel(Path(r["source_path"]), opts.raw_dir)

    stats = report.compute_statistics(rows, rejected, classes, opts)
    _write_outputs(rows, rejected, stats, opts)
    return BuildResult(rows, rejected, stats)


def scan(opts: BuildOptions) -> dict[str, Counter]:
    """Count candidate files per class and source without decoding any audio.

    Useful before a build to see which classes still need more data.
    """
    classes = load_classes()
    candidates = collect(opts.raw_dir, classes, load_label_map(classes), opts.sources)
    counts: dict[str, Counter] = {c.name: Counter() for c in classes}
    groups: dict[str, set[str]] = {c.name: set() for c in classes}
    for c in candidates:
        counts[c.class_name][c.source] += 1
        groups[c.class_name].add(c.origin_id)
    for name, counter in counts.items():
        counter["unique_recordings"] = len(groups[name])
    return counts


def _assign_audio_ids(chosen: list[Selected], sound_class: SoundClass) -> None:
    """IDs like GUN-00001, numbered in split order then source path, so they are stable."""
    order = {s: i for i, s in enumerate(SPLITS)}
    chosen.sort(key=lambda s: (order[s.split], s.candidate.source, s.candidate.path.as_posix()))
    for n, s in enumerate(chosen, start=1):
        s.audio_id = f"{sound_class.code}-{n:05d}"


def _check_no_leakage(selected) -> None:
    split_of: dict[str, str] = {}
    for s in selected:
        previous = split_of.setdefault(s.candidate.group_id, s.split)
        if previous != s.split:
            raise AssertionError(f"group {s.candidate.group_id} is in both {previous} and {s.split}")


def _materialize(s: Selected, slug: str, opts: BuildOptions) -> dict:
    c, info = s.candidate, s.info
    rel_dest = Path(s.split) / slug / f"{s.audio_id}{c.path.suffix.lower()}"
    dest = opts.out_dir / rel_dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _copy(c.path, dest, opts.hardlink)
    return {
        "audio_id": s.audio_id,
        "filename": rel_dest.as_posix(),
        "class_name": c.class_name,
        "class_slug": slug,
        "split": s.split,
        "source_dataset": c.source,
        "source_label": c.source_label,
        "source_path": _rel(c.path, opts.raw_dir),
        "original_filename": c.path.name,
        "origin_id": c.origin_id,
        "group_id": c.group_id,
        "freesound_id": c.freesound_id,
        "license": c.license,
        "format": info.format,
        "duration_s": info.duration_s,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "bit_depth": info.bit_depth or "",
        "file_size_bytes": info.file_size_bytes,
        "sha256": info.sha256,
        "peak_amplitude": info.peak_amplitude,
        "quality_grade": s.quality.grade,
        "quality_issues": "; ".join(s.quality.issues),
        "environment": c.environment,
        "device": c.device,
        "distance_m": c.distance_m,
        "speaker": c.speaker,
        "is_augmented": False,
        "parent_audio_id": "",
        "notes": c.notes,
    }


def _copy(src: Path, dest: Path, hardlink: bool) -> None:
    if hardlink:
        try:
            os.link(src, dest)
            return
        except OSError:
            pass  # different drive or unsupported filesystem: fall back to copying
    shutil.copy2(src, dest)


def _prepare_output(opts: BuildOptions) -> None:
    existing = [opts.out_dir / s for s in SPLITS if (opts.out_dir / s).exists()]
    if existing and not opts.force:
        raise FileExistsError(
            f"{opts.out_dir} already contains a built dataset; pass --force to rebuild it"
        )
    for split_dir in existing:
        shutil.rmtree(split_dir)


def _write_outputs(rows, rejected, stats, opts: BuildOptions) -> None:
    opts.metadata_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(opts.metadata_dir / "dataset_metadata.csv", METADATA_FIELDS, rows)
    _write_csv(opts.metadata_dir / "rejected_files.csv", REJECTED_FIELDS, rejected)
    report.write_statistics(stats, opts.metadata_dir / "dataset_statistics.json")


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rejection(c: Candidate, reason: str) -> dict:
    return {
        "source_dataset": c.source,
        "source_path": c.path.as_posix(),
        "class_name": c.class_name,
        "group_id": c.group_id,
        "reason": reason,
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
