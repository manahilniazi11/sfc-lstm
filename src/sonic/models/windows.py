"""Clip-level evaluation data: every window of a recording, cut the way the web app cuts it.

Training segments keep only the event windows of each clip. The web app
instead analyses *every* 1 s window (inference mode) and combines them into
one decision, so validation and test clips are evaluated that way too.
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dataset.config import DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR
from ..features import extract, load_feature_settings
from ..features.build import DEFAULT_FEATURES_DIR
from ..preprocessing.pipeline import INFERENCE, preprocess, preprocess_file
from ..preprocessing.settings import load_settings
from .data import cache_key


@dataclass
class ClipWindows:
    """Windows of many clips, stacked; clip ``i`` owns rows ``offsets[i]:offsets[i + 1]``."""

    audio_ids: np.ndarray  # (clips,)
    labels: np.ndarray  # (clips,) class index
    sources: np.ndarray  # (clips,) source dataset
    offsets: np.ndarray  # (clips + 1,)
    X: np.ndarray  # (windows, 123)
    logmel: np.ndarray  # (windows, 64, 63) float16
    wave: np.ndarray  # (windows, 16000) float16

    def batch(self) -> dict[str, np.ndarray]:
        return {"X": self.X, "logmel": self.logmel, "wave": self.wave}


def window_features(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference-mode windows of one recording -> (vectors, log-mels, waveforms), silent windows dropped."""
    settings, fsettings = load_settings(), load_feature_settings()
    segments = [s for s in preprocess(audio, sr, settings, INFERENCE) if not s.silent]
    return stack_features(segments, settings.sample_rate, fsettings)


def stack_features(segments, sr, fsettings):
    if not segments:
        return np.zeros((0, 123), np.float32), np.zeros((0, fsettings.n_mels, 63), np.float16), np.zeros((0, sr), np.float16)
    feats = [extract(s.samples, sr, fsettings) for s in segments]
    return (
        np.stack([f.vector for f in feats]),
        np.stack([f.log_mel for f in feats]).astype(np.float16),
        np.stack([s.samples for s in segments]).astype(np.float16),
    )


def _clip_job(path: Path):
    settings, fsettings = load_settings(), load_feature_settings()
    segments = [s for s in preprocess_file(path, settings, INFERENCE) if not s.silent]
    return stack_features(segments, settings.sample_rate, fsettings)


def load_windows(
    split: str,
    classes: list[str],
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    dataset_dir: Path = DEFAULT_OUT_DIR,
    cache_dir: Path = DEFAULT_FEATURES_DIR,
    workers: int = 4,
) -> ClipWindows:
    """Windows for every clip of ``split``, computed once and cached as ``windows_<split>.npz``."""
    cache = cache_dir / f"windows_{split}.npz"
    key = cache_key()
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if str(data["fingerprint"]) == key:
            return _from_arrays(data, classes)

    with open(metadata_dir / "dataset_metadata.csv", newline="", encoding="utf-8") as f:
        clips = [r for r in csv.DictReader(f) if r["split"] == split]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_clip_job, [dataset_dir / c["filename"] for c in clips], chunksize=8))

    counts = [len(r[0]) for r in results]
    arrays = {
        "fingerprint": np.array(key),
        "audio_ids": np.array([c["audio_id"] for c in clips]),
        "class_names": np.array([c["class_name"] for c in clips]),
        "sources": np.array([c["source_dataset"] for c in clips]),
        "offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "X": np.concatenate([r[0] for r in results]),
        "logmel": np.concatenate([r[1] for r in results]),
        "wave": np.concatenate([r[2] for r in results]),
    }
    np.savez(cache, **arrays)
    return _from_arrays(arrays, classes)


def _from_arrays(data, classes: list[str]) -> ClipWindows:
    index = {c: i for i, c in enumerate(classes)}
    names = np.asarray(data["class_names"])
    keep = np.array([n in index for n in names])  # clips of classes the model knows
    offsets = np.asarray(data["offsets"])
    rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in np.flatnonzero(keep)] or [np.array([], int)])
    counts = np.diff(offsets)[keep]
    return ClipWindows(
        audio_ids=np.asarray(data["audio_ids"])[keep],
        labels=np.array([index[n] for n in names[keep]]),
        sources=np.asarray(data["sources"])[keep],
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        X=np.asarray(data["X"])[rows],
        logmel=np.asarray(data["logmel"])[rows],
        wave=np.asarray(data["wave"])[rows],
    )
