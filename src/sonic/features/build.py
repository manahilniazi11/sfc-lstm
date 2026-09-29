"""Extract features for every segment (original and augmented) into ``<data>/features/``.

Outputs, all in the same row order:

* ``features.csv``: one row per segment: identifiers, class, split, whether it
  is augmented, then the 123 summary features (classical models).
* ``logmel.npy``: float16 array (rows, n_mels, frames), the CNN input.
* ``feature_info.json``: feature names and the settings used, so a trained
  model can check it is fed the features it was trained on.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from ..augmentation.build import DEFAULT_AUGMENTED_DIR
from ..dataset.config import AUDIO_DATA_DIR, DEFAULT_METADATA_DIR
from ..preprocessing.build import DEFAULT_SEGMENTS_DIR
from .extract import extract, feature_names
from .settings import FeatureSettings, load_feature_settings

DEFAULT_FEATURES_DIR = AUDIO_DATA_DIR / "features"
ID_FIELDS = ["segment_id", "audio_id", "class_name", "split", "is_augmented", "source_dataset"]


def load_rows(metadata_dir: Path, segments_dir: Path, augmented_dir: Path) -> list[tuple[dict, Path]]:
    """Original segments of every split, plus augmented training segments."""
    rows = []
    with open(metadata_dir / "segments.csv", newline="", encoding="utf-8") as f:
        rows += [({**r, "is_augmented": False}, segments_dir / r["filename"]) for r in csv.DictReader(f)]
    augmented = metadata_dir / "augmented_segments.csv"
    if augmented.exists():
        with open(augmented, newline="", encoding="utf-8") as f:
            rows += [({**r, "is_augmented": True}, augmented_dir / r["filename"]) for r in csv.DictReader(f)]
    return rows


def _extract_file(job: tuple[Path, FeatureSettings]) -> tuple[np.ndarray, np.ndarray]:
    path, settings = job
    y, sr = sf.read(path, dtype="float32")
    features = extract(y, sr, settings)
    return features.vector, features.log_mel.astype(np.float16)


def build_features(
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    segments_dir: Path = DEFAULT_SEGMENTS_DIR,
    augmented_dir: Path = DEFAULT_AUGMENTED_DIR,
    out_dir: Path = DEFAULT_FEATURES_DIR,
    workers: int = 4,
) -> dict:
    settings = load_feature_settings()
    names = feature_names(settings)
    rows = load_rows(metadata_dir, segments_dir, augmented_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors, mels = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(path, settings) for _, path in rows]
        for i, (vector, mel) in enumerate(pool.map(_extract_file, jobs, chunksize=64), start=1):
            vectors.append(vector)
            mels.append(mel)
            if i % 1000 == 0 or i == len(jobs):
                print(f"\r  {i}/{len(jobs)} segments", end="", flush=True)
    print()

    with open(out_dir / "features.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ID_FIELDS + names)
        for (row, _), vector in zip(rows, vectors):
            writer.writerow([row[k] for k in ID_FIELDS] + [f"{v:.6g}" for v in vector])
    np.save(out_dir / "logmel.npy", np.stack(mels))
    (out_dir / "feature_info.json").write_text(
        json.dumps({"feature_names": names, "settings": asdict(settings), "rows": len(rows)}, indent=1),
        encoding="utf-8",
    )
    matrix = np.stack(vectors)
    return {
        "rows": len(rows),
        "features": len(names),
        "mel_shape": mels[0].shape,
        "non_finite": int((~np.isfinite(matrix)).sum()),
        "out_dir": out_dir,
    }
