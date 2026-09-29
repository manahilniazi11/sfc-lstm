"""One WAV of **test-split** clips for live-monitoring demos (`python -m sonic.demo playlist`).

Playing it on the PC while live monitoring listens to Stereo Mix (or a
virtual cable) tests the whole live path on recordings no model was trained
on. Each clip is followed by a short silence so detections do not run into
each other, and `live_playlist_manifest.csv` says which clip plays when, so
the live results can be checked against the real labels.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from ..dataset.config import AUDIO_DATA_DIR, DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR
from ..preprocessing import steps

DEFAULT_DEMO_DIR = AUDIO_DATA_DIR / "demo"
PLAYBACK_SR = 48000
MAX_CLIP_S = 8.0
GAP_S = 3.0
PEAK = 0.7  # playback level: loud enough for Stereo Mix, below clipping


def build_playlist(
    per_class: int = 2,
    out_dir: Path = DEFAULT_DEMO_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    dataset_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 7,
) -> list[dict]:
    with open(metadata_dir / "dataset_metadata.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["class_name"]].append(r)
    rng = random.Random(seed)
    picks = []
    for name in sorted(by_class):
        clips = sorted(by_class[name], key=lambda r: r["audio_id"])
        rng.shuffle(clips)
        picks.extend(clips[:per_class])
    rng.shuffle(picks)  # mixed order, like real life

    gap = np.zeros(int(GAP_S * PLAYBACK_SR), np.float32)
    pieces, manifest, t = [gap], [], GAP_S
    for r in picks:
        audio, sr = steps.load(dataset_dir / r["filename"])
        y = steps.resample(steps.to_mono(audio), sr, PLAYBACK_SR)[: int(MAX_CLIP_S * PLAYBACK_SR)]
        peak = float(np.abs(y).max()) or 1.0
        y = (y / peak * PEAK).astype(np.float32)
        manifest.append({"start_s": round(t, 2), "end_s": round(t + y.size / PLAYBACK_SR, 2), "audio_id": r["audio_id"],
                         "class_name": r["class_name"], "source_dataset": r["source_dataset"]})
        pieces += [y, gap]
        t += y.size / PLAYBACK_SR + GAP_S

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "live_test_playlist.wav", np.concatenate(pieces), PLAYBACK_SR, subtype="PCM_16")
    with open(out_dir / "live_playlist_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest
