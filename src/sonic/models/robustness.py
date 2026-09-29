"""Hidden-test simulator (SRS 1.8 item 9) and the factory-shortcut test.

The SRS says hidden tests may contain background noise, echo, low volume,
different devices, partial events, re-encoded files and distant sources.
Each condition below degrades every *test* clip, then runs it through the
same preprocessing and feature extraction as the web app.

Background noise comes from the Background Noise **test** clips only, which
no model saw in training or in augmentation.
"""

from __future__ import annotations

import csv
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from ..augmentation import transforms as T
from ..dataset.config import DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR, DEFAULT_RAW_DIR
from ..features.build import DEFAULT_FEATURES_DIR
from ..preprocessing import steps
from .data import cache_key
from .windows import ClipWindows, _from_arrays, window_features

SR = 16000
CACHE_DIR = DEFAULT_FEATURES_DIR / "robustness"
BACKGROUND = "Background Noise"
NORMAL = "Normal Machinery"
CONDITIONS = [
    "noise_20dB", "noise_10dB", "noise_5dB", "noise_0dB",
    "echo", "low_volume", "phone", "distant", "partial", "mp3_64k",
]
DESCRIPTIONS = {
    "noise_20dB": "background noise at 20 dB SNR", "noise_10dB": "background noise at 10 dB SNR",
    "noise_5dB": "background noise at 5 dB SNR", "noise_0dB": "background noise at 0 dB SNR (as loud as the event)",
    "echo": "strong reverb (RT60 0.8 s, 50 % wet)", "low_volume": "-30 dB quieter over a -60 dBFS microphone hiss",
    "phone": "phone microphone (300-3400 Hz) with clipping", "distant": "far-away source (low-pass 2.5 kHz, reverb)",
    "partial": "only 40-60 % of the recording", "mp3_64k": "re-encoded as low-bitrate MP3",
}


def load_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return steps.resample(steps.to_mono(audio), sr, SR)


def degrade(y: np.ndarray, condition: str, rng: np.random.Generator, noise: np.ndarray | None) -> np.ndarray:
    """Apply one hidden-test condition to a 16 kHz mono recording ("none" leaves it unchanged)."""
    if condition == "none":
        return y
    level = steps.rms_dbfs(y)
    if condition.startswith("noise_"):
        snr = float(condition.split("_")[1].removesuffix("dB"))
        return T.mix_background(y, np.resize(noise, y.size), snr, level)
    if condition == "echo":
        return T.reverb(y, SR, rt60=0.8, wet=0.5, rng=rng)
    if condition == "low_volume":
        hiss = rng.standard_normal(y.size).astype(np.float32) * np.float32(10 ** (-60 / 20))
        return (y * np.float32(10 ** (-30 / 20)) + hiss).astype(np.float32)
    if condition == "phone":
        return T.device(y, SR, (300, 3400), clip_level=0.5)
    if condition == "distant":
        return T.distance(y, SR, lowpass_hz=2500, extra_wet=0.5, rng=rng)
    if condition == "partial":
        keep = int(y.size * rng.uniform(0.4, 0.6))
        start = int(rng.integers(0, y.size - keep + 1))
        return y[start : start + keep]
    if condition == "mp3_64k":
        return mp3_roundtrip(y)
    raise ValueError(f"unknown condition {condition}")


def mp3_roundtrip(y: np.ndarray) -> np.ndarray:
    """Encode to a low-bitrate MP3 in memory and decode it again (libsndfile's MP3 encoder)."""
    buffer = io.BytesIO()
    sf.write(buffer, y, SR, format="MP3", subtype="MPEG_LAYER_III", compression_level=0.9)
    buffer.seek(0)
    decoded, _ = sf.read(buffer, dtype="float32")
    return decoded


def _job(args) -> tuple:
    path, condition, noise_path, seed = args
    rng = np.random.default_rng(seed)
    y = load_mono(path)
    noise = load_mono(noise_path) if noise_path else None
    return window_features(degrade(y, condition, rng, noise), SR)


def condition_windows(condition: str, classes: list[str], workers: int = 4) -> ClipWindows:
    """Windows of every test clip under one condition, cached per condition."""
    cache = CACHE_DIR / f"{condition}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if str(data["fingerprint"]) == cache_key():
            return _from_arrays(data, classes)
    with open(DEFAULT_METADATA_DIR / "dataset_metadata.csv", newline="", encoding="utf-8") as f:
        clips = [r for r in csv.DictReader(f) if r["split"] == "test"]
    backgrounds = [c for c in clips if c["class_name"] == BACKGROUND]
    jobs = []
    for i, c in enumerate(clips):
        noise = None
        if condition.startswith("noise_"):
            pool = [b for b in backgrounds if b["audio_id"] != c["audio_id"]]  # never mix a clip with itself
            noise = DEFAULT_OUT_DIR / pool[i % len(pool)]["filename"]
        jobs.append((DEFAULT_OUT_DIR / c["filename"], condition, noise, 1000 + i))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_job, jobs, chunksize=8))
    arrays = _pack(results, [c["audio_id"] for c in clips], [c["class_name"] for c in clips],
                   [c["source_dataset"] for c in clips])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    return _from_arrays(arrays, classes)


def factory_windows(classes: list[str], raw_dir: Path = DEFAULT_RAW_DIR, workers: int = 4) -> ClipWindows:
    """Unseen MIMII normal-operation clips (never in the dataset, training or augmentation).

    Labelled Normal Machinery when the models have that class, else Background Noise.
    """
    cache = CACHE_DIR / "factory_normal.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if str(data["fingerprint"]) == cache_key():
            return _from_arrays(data, classes)
    paths = sorted((raw_dir / "mimii_holdout").rglob("*.wav"))
    if not paths:
        raise FileNotFoundError("run: python -m sonic.dataset download mimii-valve-normal-holdout")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_job, [(p, "none", None, 0) for p in paths], chunksize=4))
    label = NORMAL if NORMAL in classes else BACKGROUND
    arrays = _pack(results, [p.stem for p in paths], [label] * len(paths), ["mimii_holdout"] * len(paths))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    return _from_arrays(arrays, classes)


def _pack(results, audio_ids, class_names, sources) -> dict:
    counts = [len(r[0]) for r in results]
    return {
        "fingerprint": np.array(cache_key()),
        "audio_ids": np.array(audio_ids),
        "class_names": np.array(class_names),
        "sources": np.array(sources),
        "offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "X": np.concatenate([r[0] for r in results]),
        "logmel": np.concatenate([r[1] for r in results]),
        "wave": np.concatenate([r[2] for r in results]),
    }
