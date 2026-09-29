"""Preprocess the built dataset into training segments.

Reads ``data/metadata/dataset_metadata.csv``, runs every clip through the
pipeline in training mode, and writes 16-bit WAV segments to
``<data>/segments/<split>/<class slug>/<audio id>_<n>.wav`` plus
``data/metadata/segments.csv``. Segments keep their clip's Audio ID and
split, as the SRS requires for derived segments.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

from ..dataset.config import AUDIO_DATA_DIR, DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR
from .pipeline import TRAINING, preprocess_file
from .settings import AudioSettings, load_settings

DEFAULT_SEGMENTS_DIR = AUDIO_DATA_DIR / "segments"
SEGMENT_FIELDS = [
    "segment_id", "audio_id", "filename", "class_name", "class_slug", "split",
    "source_dataset", "start_s", "end_s", "level_dbfs", "padded",
]


def clip_seed(audio_id: str) -> int:
    """A stable per-clip seed, so padding noise and offsets are identical on every run."""
    return int.from_bytes(hashlib.sha256(audio_id.encode()).digest()[:4], "little")


def _process(job: tuple[dict, Path, Path, AudioSettings]) -> tuple[list[dict], str]:
    row, dataset_dir, out_dir, settings = job
    try:
        segments = preprocess_file(dataset_dir / row["filename"], settings, TRAINING, clip_seed(row["audio_id"]))
    except Exception as exc:  # report and continue: one bad file must not stop the build
        return [], f"{row['audio_id']}: {exc}"
    rows = []
    for n, seg in enumerate(s for s in segments if not s.silent):
        segment_id = f"{row['audio_id']}_{n:02d}"
        rel = Path(row["split"]) / row["class_slug"] / f"{segment_id}.wav"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / rel, seg.samples, settings.sample_rate, subtype="PCM_16")
        rows.append({
            "segment_id": segment_id,
            "audio_id": row["audio_id"],
            "filename": rel.as_posix(),
            "class_name": row["class_name"],
            "class_slug": row["class_slug"],
            "split": row["split"],
            "source_dataset": row["source_dataset"],
            "start_s": seg.start_s,
            "end_s": seg.end_s,
            "level_dbfs": seg.level_dbfs,
            "padded": seg.padded,
        })
    return rows, "" if rows else f"{row['audio_id']}: no usable segment"


def build_segments(
    dataset_dir: Path = DEFAULT_OUT_DIR,
    out_dir: Path = DEFAULT_SEGMENTS_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    workers: int = 4,
) -> dict:
    settings = load_settings()
    with open(metadata_dir / "dataset_metadata.csv", newline="", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    if out_dir.exists():
        shutil.rmtree(out_dir)  # segments are fully derived from the dataset; rebuild from scratch

    rows, problems = [], []
    jobs = [(clip, dataset_dir, out_dir, settings) for clip in clips]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (clip_rows, problem) in enumerate(pool.map(_process, jobs, chunksize=16), start=1):
            rows.extend(clip_rows)
            if problem:
                problems.append(problem)
            if i % 250 == 0 or i == len(jobs):
                print(f"\r  {i}/{len(jobs)} clips -> {len(rows)} segments", end="", flush=True)
    print()

    rows.sort(key=lambda r: r["segment_id"])
    with open(metadata_dir / "segments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["class_name"]][r["split"]] += 1
    return {"settings": settings, "counts": counts, "segments": len(rows), "problems": problems}
