"""Training data for the models: segment features, labels and splits."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..dataset.config import DEFAULT_METADATA_DIR, load_classes
from ..features import feature_names, load_feature_settings
from ..features.build import DEFAULT_FEATURES_DIR
from ..preprocessing.settings import load_settings


@dataclass
class SegmentData:
    """All segments (original and augmented) with their features, in features.csv row order."""

    X: np.ndarray  # (n, 123) float32 summary features
    logmel: np.ndarray  # (n, 64, 63) float16, memory-mapped
    y: np.ndarray  # (n,) class index into ``classes``
    split: np.ndarray
    is_augmented: np.ndarray
    audio_id: np.ndarray
    segment_id: np.ndarray
    source: np.ndarray
    classes: list[str]

    def rows(self, split: str, augmented: bool | None = None) -> np.ndarray:
        """Row indices of a split; ``augmented=False`` keeps only original segments."""
        mask = self.split == split
        if augmented is not None:
            mask &= self.is_augmented == augmented
        return np.flatnonzero(mask)


def model_classes(present: set[str]) -> list[str]:
    """Classes in config order, limited to those that have training data.

    Person Asking for Help has no recordings yet, so today's models have 9
    classes; once recordings exist the same code trains 10.
    """
    return [c.name for c in load_classes() if c.name in present]


def critical_classes() -> list[str]:
    return [c.name for c in load_classes() if c.critical]


def fingerprint() -> str:
    """Hash of everything that shapes the model input: preprocessing + feature settings + feature names."""
    settings = load_feature_settings()
    blob = json.dumps(
        {"audio": asdict(load_settings()), "features": asdict(settings), "names": feature_names(settings)},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def cache_key() -> str:
    """Fingerprint of the model input *and* the data it was built from.

    Caches (app-mode windows, YAMNet embeddings, robustness conditions) use
    this, so a rebuilt dataset or new augmentation never reuses stale data.
    """
    digest = hashlib.sha256(fingerprint().encode())
    for name in ("dataset_metadata.csv", "segments.csv", "augmented_segments.csv"):
        path = DEFAULT_METADATA_DIR / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def load_segments(features_dir: Path = DEFAULT_FEATURES_DIR) -> SegmentData:
    with open(features_dir / "features.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    n_id = header.index(feature_names(load_feature_settings())[0])
    ids = np.array([r[:n_id] for r in rows])
    X = np.array([r[n_id:] for r in rows], dtype=np.float32)
    col = {name: i for i, name in enumerate(header[:n_id])}

    class_names = ids[:, col["class_name"]]
    split = ids[:, col["split"]]
    classes = model_classes(set(class_names[split == "train"]))
    index = {name: i for i, name in enumerate(classes)}
    keep = np.array([c in index for c in class_names])  # drop classes without training data

    logmel = np.load(features_dir / "logmel.npy", mmap_mode="r")
    if logmel.shape[0] != X.shape[0]:
        raise ValueError("features.csv and logmel.npy have different row counts; rebuild features")
    return SegmentData(
        X=X[keep],
        logmel=logmel[keep] if not keep.all() else logmel,
        y=np.array([index[c] for c in class_names[keep]]),
        split=split[keep],
        is_augmented=ids[keep, col["is_augmented"]] == "True",
        audio_id=ids[keep, col["audio_id"]],
        segment_id=ids[keep, col["segment_id"]],
        source=ids[keep, col["source_dataset"]],
        classes=classes,
    )
