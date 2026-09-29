"""Shortcut check: can the class be predicted from recording format alone?

Each class comes mostly from one source dataset, and datasets differ in
sample rate, channel count, bandwidth, padding and level. A model can learn
those instead of the sound. Two checks train a small classifier on *only*
such nuisance features:

1. Predict the class. Some accuracy above chance is expected, because
   brightness and impulsiveness are real properties of e.g. glass breaking.
2. Within each class, predict the source dataset. The sounds are the same
   kind, so accuracy above the majority guess means a dataset fingerprint.

Run: ``python -m sonic.preprocessing check`` (raw clips vs processed segments).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from ..dataset.config import DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR
from .build import DEFAULT_SEGMENTS_DIR
from .steps import rms_dbfs, to_mono

FEATURES = ["sample_rate", "channels", "hf_energy_ratio", "zero_fraction", "level_dbfs"]


def nuisance_features(path: Path) -> list[float]:
    """Properties of the recording, not of the sound event."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    y = to_mono(audio)
    spectrum = np.abs(np.fft.rfft(y[: sr * 10])) ** 2  # first 10 s is plenty
    freqs = np.fft.rfftfreq(min(y.size, sr * 10), 1 / sr)
    hf = spectrum[freqs >= 7000].sum() / (spectrum.sum() + 1e-12)
    return [sr, audio.shape[1], float(hf), float(np.mean(y == 0)), rms_dbfs(y)]


def run_check(rows: list[dict], root: Path, per_class: int = 60, seed: int = 0) -> tuple[float, float]:
    """Cross-validated accuracy of a nuisance-feature classifier, and the chance level."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["class_name"], []).append(r)
    sample = []
    for items in by_class.values():
        idx = rng.choice(len(items), size=min(per_class, len(items)), replace=False)
        sample.extend(items[i] for i in idx)

    X = np.array([nuisance_features(root / r["filename"]) for r in sample])
    y = np.array([r["class_name"] for r in sample])
    groups = np.array([r["audio_id"] for r in sample])  # segments of one clip stay in one fold
    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    scores = cross_val_score(model, X, y, cv=GroupKFold(n_splits=5), groups=groups)
    return float(scores.mean()), 1 / len(by_class)


def source_check(rows: list[dict], root: Path, per_source: int = 40, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Per class with 2+ sources: can nuisance features tell which dataset a clip came from?

    Within one class the sounds are the same kind, so any accuracy above the
    majority-guess baseline comes from dataset fingerprints. Returns
    ``{class: (accuracy, majority baseline)}``.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score

    rng = np.random.default_rng(seed)
    results = {}
    for class_name in sorted({r["class_name"] for r in rows}):
        by_source: dict[str, list[dict]] = {}
        for r in rows:
            if r["class_name"] == class_name:
                by_source.setdefault(r["source_dataset"], []).append(r)
        by_source = {s: items for s, items in by_source.items() if len(items) >= 20}
        if len(by_source) < 2:
            continue
        sample = []
        for items in by_source.values():
            idx = rng.choice(len(items), size=min(per_source, len(items)), replace=False)
            sample.extend(items[i] for i in idx)
        X = np.array([nuisance_features(root / r["filename"]) for r in sample])
        y = np.array([r["source_dataset"] for r in sample])
        groups = np.array([r["audio_id"] for r in sample])
        model = RandomForestClassifier(n_estimators=200, random_state=seed)
        accuracy = cross_val_score(model, X, y, cv=GroupKFold(n_splits=5), groups=groups).mean()
        baseline = max(np.unique(y, return_counts=True)[1]) / len(y)
        results[class_name] = (float(accuracy), float(baseline))
    return results


def main_check(metadata_dir: Path = DEFAULT_METADATA_DIR) -> None:
    for label, csv_name, root in [
        ("raw clips", "dataset_metadata.csv", DEFAULT_OUT_DIR),
        ("processed segments", "segments.csv", DEFAULT_SEGMENTS_DIR),
    ]:
        path = metadata_dir / csv_name
        if not path.exists():
            print(f"{label}: {csv_name} not found, skipped")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        accuracy, chance = run_check(rows, root)
        print(f"\n{label}\n  class from format only: {accuracy:6.1%}   (chance {chance:.1%})")
        for class_name, (acc, base) in source_check(rows, root).items():
            print(f"  {class_name:<20} which dataset? {acc:6.1%}   (majority guess {base:.1%})")
