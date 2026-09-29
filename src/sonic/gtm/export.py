"""Training audio for the Google Teachable Machine (GTM) audio project (SRS Steps 5 and 9).

GTM audio projects cannot import audio files: samples can only be recorded
through a microphone in the browser. To train GTM on *the same training
recordings* as the Python model (SRS Step 9), each class's training segments
are joined into one continuous WAV per class; the file is played into a
virtual audio cable that the browser uses as its microphone, while GTM
records. GTM cuts what it hears into 1 s samples.

Only **training-split** segments are used (never validation or test), and
``manifest.csv`` records which Audio ID plays at which second of which
file, as evidence for the report.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from ..dataset.config import AUDIO_DATA_DIR, DEFAULT_METADATA_DIR, load_classes
from ..preprocessing.build import DEFAULT_SEGMENTS_DIR
from ..preprocessing.steps import resample

DEFAULT_GTM_DIR = AUDIO_DATA_DIR / "gtm_export"
PLAYBACK_SR = 48000  # the rate Windows and browsers use for microphones by default
SEGMENT_S = 1.0


def pick_segments(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """``count`` segments of one class, spread over as many recordings (Audio IDs) as possible."""
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_clip[r["audio_id"]].append(r)
    clips = sorted(by_clip)
    rng.shuffle(clips)
    for c in clips:
        rng.shuffle(by_clip[c])
    picked: list[dict] = []
    while len(picked) < count and any(by_clip.values()):
        for c in clips:  # one segment from every recording before a second from any
            if by_clip[c] and len(picked) < count:
                picked.append(by_clip[c].pop())
    return picked


def export(
    seconds_per_class: int = 150,
    out_dir: Path = DEFAULT_GTM_DIR,
    segments_dir: Path = DEFAULT_SEGMENTS_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    seed: int = 42,
) -> dict[str, int]:
    """Write ``<class slug>.wav`` (one 1 s training segment per second) and ``manifest.csv``."""
    with open(metadata_dir / "segments.csv", newline="", encoding="utf-8") as f:
        train = [r for r in csv.DictReader(f) if r["split"] == "train"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        by_class[r["class_name"]].append(r)

    manifest, counts = [], {}
    for sound_class in load_classes():
        rows = by_class.get(sound_class.name)
        if not rows:
            continue
        picked = pick_segments(rows, seconds_per_class, rng)
        audio = []
        for second, r in enumerate(picked):
            y, sr = sf.read(segments_dir / r["filename"], dtype="float32")
            audio.append(resample(y, sr, PLAYBACK_SR))
            manifest.append({"class_name": sound_class.name, "file": f"{sound_class.slug}.wav",
                             "start_s": second * SEGMENT_S, "segment_id": r["segment_id"],
                             "audio_id": r["audio_id"], "split": r["split"], "source_dataset": r["source_dataset"]})
        sf.write(out_dir / f"{sound_class.slug}.wav", np.concatenate(audio), PLAYBACK_SR, subtype="PCM_16")
        counts[sound_class.name] = len(picked)

    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return counts


def add(
    slug: str,
    seconds: int = 150,
    source: str | None = None,
    out_dir: Path = DEFAULT_GTM_DIR,
    segments_dir: Path = DEFAULT_SEGMENTS_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    seed: int = 42,
) -> tuple[Path, int]:
    """Write one more playback file for an existing GTM project, leaving every other file as it is.

    Used to add a class to a trained project (``help_request``) or extra
    samples to a class (``background_noise`` with ``source="edge-tts"``: only
    the synthetic ordinary sentences). The file is ``<slug>.wav``, or
    ``<slug>_<source>.wav`` with a source; its rows replace any earlier rows
    for that file in ``manifest.csv``.
    """
    sound_class = next(c for c in load_classes() if c.slug == slug)
    with open(metadata_dir / "segments.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train" and r["class_name"] == sound_class.name
                and (source is None or r["source_dataset"] == source)]
    if not rows:
        raise ValueError(f"no training segments for {sound_class.name}" + (f" from {source}" if source else ""))
    name = f"{slug}_{source.replace('-', '')}.wav" if source else f"{slug}.wav"
    picked = pick_segments(rows, seconds, random.Random(f"{seed}:{name}"))  # own generator: other files unaffected
    audio, added = [], []
    for second, r in enumerate(picked):
        y, sr = sf.read(segments_dir / r["filename"], dtype="float32")
        audio.append(resample(y, sr, PLAYBACK_SR))
        added.append({"class_name": sound_class.name, "file": name, "start_s": second * SEGMENT_S,
                      "segment_id": r["segment_id"], "audio_id": r["audio_id"], "split": r["split"],
                      "source_dataset": r["source_dataset"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / name, np.concatenate(audio), PLAYBACK_SR, subtype="PCM_16")

    manifest_path = out_dir / "manifest.csv"
    manifest = []
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            manifest = [r for r in csv.DictReader(f) if r["file"] != name]
    manifest += added
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(added[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return out_dir / name, len(picked)
