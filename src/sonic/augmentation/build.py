"""Generate augmented training segments and ``data/metadata/augmented_segments.csv``.

Only training segments are augmented, and the background-noise pool is
built from training segments of Background Noise and Normal Machinery. Each copy keeps its parent's Audio ID
and split (SRS Hint), is marked ``is_augmented``, and never counts as a
unique clip. Classes are topped up to the same training size.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from ..dataset.config import AUDIO_DATA_DIR, DEFAULT_METADATA_DIR
from ..preprocessing.build import DEFAULT_SEGMENTS_DIR
from ..preprocessing.settings import load_settings
from .chain import augment, load_config

DEFAULT_AUGMENTED_DIR = AUDIO_DATA_DIR / "augmented"
NOISE_CLASSES = ("Background Noise", "Normal Machinery")
FIELDS = [
    "segment_id", "parent_segment_id", "audio_id", "filename", "class_name", "class_slug",
    "split", "source_dataset", "is_augmented", "augmentation", "noise_source",
]


def seed_for(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def plan_copies(originals: list[dict], target: int, max_copies: int, rng: np.random.Generator) -> list[tuple[dict, int]]:
    """Spread ``target - len(originals)`` copies as evenly as possible over the originals."""
    need = max(0, target - len(originals))
    if need == 0 or not originals:
        return []
    base, extra = divmod(need, len(originals))
    order = rng.permutation(len(originals))
    counts = np.full(len(originals), base)
    counts[order[:extra]] += 1
    counts = np.minimum(counts, max_copies)
    return [(originals[i], k) for i in range(len(originals)) for k in range(int(counts[i]))]


def build_noise_bank(train_rows: list[dict], segments_dir: Path, bank_dir: Path) -> int:
    """Stack all background sources into one array file the worker processes memory-map.

    Sources: *training* segments of Background Noise and Normal Machinery
    (factory recordings of machines working normally). Only training
    segments are used, so no validation or test audio leaks into training.
    """
    arrays, ids, audio_ids = [], [], []
    for r in train_rows:
        if r["class_name"] in NOISE_CLASSES:
            y, _ = sf.read(segments_dir / r["filename"], dtype="float32")
            arrays.append(y)
            ids.append(r["segment_id"])
            audio_ids.append(r["audio_id"])
    if not arrays:
        raise ValueError(f"no training segments of {NOISE_CLASSES} to use as background noise")
    bank_dir.mkdir(parents=True, exist_ok=True)
    np.save(bank_dir / "noise_bank.npy", np.stack(arrays).astype(np.float32))
    (bank_dir / "noise_bank.json").write_text(json.dumps({"ids": ids, "audio_ids": audio_ids}), encoding="utf-8")
    return len(ids)


_worker: dict = {}


def _init_worker(bank_dir: str) -> None:
    _worker["bank"] = np.load(Path(bank_dir) / "noise_bank.npy", mmap_mode="r")
    meta = json.loads((Path(bank_dir) / "noise_bank.json").read_text(encoding="utf-8"))
    _worker["ids"], _worker["audio_ids"] = meta["ids"], meta["audio_ids"]
    _worker["cfg"], _worker["settings"] = load_config(), load_settings()


def _make_copy(job: tuple[dict, int, str, str]) -> dict:
    parent, k, segments_dir, out_dir = job
    segment_id = f"{parent['segment_id']}_aug{k}"
    rng = np.random.default_rng(seed_for(segment_id))
    bank, ids, audio_ids = _worker["bank"], _worker["ids"], _worker["audio_ids"]
    while True:  # never mix a background clip with itself
        i = int(rng.integers(len(ids)))
        if audio_ids[i] != parent["audio_id"]:
            break
    y, _ = sf.read(Path(segments_dir) / parent["filename"], dtype="float32")
    y, description = augment(y, np.asarray(bank[i]), _worker["cfg"], _worker["settings"], rng)
    rel = Path("train") / parent["class_slug"] / f"{segment_id}.wav"
    (Path(out_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
    sf.write(Path(out_dir) / rel, y, _worker["settings"].sample_rate, subtype="PCM_16")
    return {
        "segment_id": segment_id,
        "parent_segment_id": parent["segment_id"],
        "audio_id": parent["audio_id"],
        "filename": rel.as_posix(),
        "class_name": parent["class_name"],
        "class_slug": parent["class_slug"],
        "split": "train",
        "source_dataset": parent["source_dataset"],
        "is_augmented": True,
        "augmentation": description,
        "noise_source": ids[i],
    }


def build_augmented(
    segments_dir: Path = DEFAULT_SEGMENTS_DIR,
    out_dir: Path = DEFAULT_AUGMENTED_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    workers: int = 4,
    seed: int = 42,
) -> dict:
    cfg = load_config()
    with open(metadata_dir / "segments.csv", newline="", encoding="utf-8") as f:
        train = [r for r in csv.DictReader(f) if r["split"] == "train"]
    if out_dir.exists():
        shutil.rmtree(out_dir)

    bank_size = build_noise_bank(train, segments_dir, out_dir / "_noise_bank")
    print(f"  noise bank: {bank_size} background segments")

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        by_class[r["class_name"]].append(r)
    jobs = []
    for class_name in sorted(by_class):
        for parent, k in plan_copies(by_class[class_name], cfg["target_train_segments_per_class"],
                                     cfg["max_copies_per_segment"], rng):
            jobs.append((parent, k, str(segments_dir), str(out_dir)))

    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(out_dir / "_noise_bank"),)) as pool:
        for i, row in enumerate(pool.map(_make_copy, jobs, chunksize=32), start=1):
            rows.append(row)
            if i % 500 == 0 or i == len(jobs):
                print(f"\r  {i}/{len(jobs)} augmented segments", end="", flush=True)
    print()

    rows.sort(key=lambda r: r["segment_id"])
    with open(metadata_dir / "augmented_segments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    originals = Counter(r["class_name"] for r in train)
    copies = Counter(r["class_name"] for r in rows)
    return {"originals": originals, "copies": copies, "bank": bank_size}
