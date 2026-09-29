"""FSD50K (zenodo.org/record/4060432): 51,197 Freesound clips, 200 AudioSet labels.

Clips are multi-label and labels are propagated up the AudioSet ontology, so
a clip is only used when its labels point at exactly one of our classes (see
:func:`classify`).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from ..candidates import Candidate
from ._util import find_file, license_from_url, read_csv

log = logging.getLogger(__name__)

# (ground-truth csv, audio folder, clip-info json)
PARTS = [
    ("dev.csv", "FSD50K.dev_audio", "dev_clips_info_FSD50K.json"),
    ("eval.csv", "FSD50K.eval_audio", "eval_clips_info_FSD50K.json"),
]


def classify(labels: list[str], label_map: dict) -> tuple[str | None, str]:
    """Pick our class for one clip: ``(class_name, "")`` or ``(None, reason)``.

    - Any label in ``_exclude_if_present`` rejects the clip (e.g. "Alarm" also
      covers telephones and doorbells).
    - Labels in ``_generic_labels`` are parents in the ontology; they only
      decide the class when no more specific mapped label is present, so
      "Alarm" + "Vehicle horn" is a Vehicle Horn clip, not an ambiguous one.
    - Clips that still point at two classes are skipped as ambiguous.
    """
    if set(label_map.get("_exclude_if_present") or []).intersection(labels):
        return None, "excluded"
    generic = set(label_map.get("_generic_labels") or [])
    mapped = [l for l in labels if not l.startswith("_") and label_map.get(l)]
    classes = {label_map[l] for l in mapped if l not in generic} or {label_map[l] for l in mapped}
    if len(classes) > 1:
        return None, "ambiguous"
    if not classes:
        return None, "unmapped"
    return classes.pop(), ""


def classified_clips(root: Path, label_map: dict) -> Iterator[tuple[str, str, str, list[str]]]:
    """Yield ``(audio folder name, freesound id, class name, labels)`` from the ground truth."""
    skipped: Counter = Counter()
    for csv_name, audio_name, _ in PARTS:
        gt = find_file(root, csv_name)
        if gt is None:
            continue
        for row in read_csv(gt):
            labels = row["labels"].split(",")
            class_name, reason = classify(labels, label_map)
            if class_name:
                yield audio_name, row["fname"], class_name, labels
            elif reason != "unmapped":
                skipped[reason] += 1
    for reason, n in skipped.items():
        log.info("fsd50k: skipped %d %s clips", n, reason)


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    _warn_unknown_labels(root, label_map)
    audio_dirs = {name: next((d for d in root.rglob(name) if d.is_dir()), None) for _, name, _ in PARTS}
    licenses = {name: _load_licenses(root, info) for _, name, info in PARTS if audio_dirs[name]}
    for audio_name, fs_id, class_name, labels in classified_clips(root, label_map):
        audio_dir = audio_dirs[audio_name]
        if audio_dir is None:
            continue
        path = audio_dir / f"{fs_id}.wav"
        if not path.exists():
            continue  # partial download: only some clips were fetched
        yield Candidate(
            source="fsd50k",
            path=path,
            class_name=class_name,
            source_label="|".join(l for l in labels if label_map.get(l) == class_name),
            group_id=f"freesound:{fs_id}",
            freesound_id=fs_id,
            license=licenses[audio_name].get(fs_id, "see FSD50K clip info"),
        )


def _load_licenses(root: Path, info_name: str) -> dict[str, str]:
    info = find_file(root, info_name)
    if info is None:
        return {}
    data = json.loads(info.read_text(encoding="utf-8"))
    return {fs_id: license_from_url(meta.get("license", "")) for fs_id, meta in data.items()}


def _warn_unknown_labels(root: Path, label_map: dict[str, str | None]) -> None:
    """Catch typos in the label map against FSD50K's own vocabulary."""
    vocab_file = find_file(root, "vocabulary.csv")
    if vocab_file is None:
        return
    # vocabulary.csv has no header: index,label,mid
    vocab = {line.split(",")[1] for line in vocab_file.read_text(encoding="utf-8").splitlines() if line}
    configured = {k for k in label_map if not k.startswith("_")}
    configured |= set(label_map.get("_exclude_if_present") or [])
    configured |= set(label_map.get("_generic_labels") or [])
    for label in sorted(configured - vocab):
        log.warning("fsd50k: label '%s' in dataset_label_map.json is not in FSD50K's vocabulary", label)
