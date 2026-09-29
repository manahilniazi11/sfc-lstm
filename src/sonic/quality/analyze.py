"""Audio quality analysis (SRS Step 13; FR xii silence, xiii clipping, xiv noise, xxxvii grading).

``assess`` runs seven checks on raw audio (before any normalization) and
grades the recording Good / Acceptable / Poor / Unusable. Each check returns
a grade and, when it is not Good, a human-readable issue; the recording's
grade is the worst of them, so the issues explain the grade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..dataset.config import CONFIG_DIR
from ..preprocessing.steps import to_mono

GRADES = ["Good", "Acceptable", "Poor", "Unusable"]
GOOD, ACCEPTABLE, POOR, UNUSABLE = range(4)
EPS = 1e-12


@dataclass
class QualityResult:
    grade: str
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.grade != "Unusable"


def load_thresholds(path: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "quality.json"
    return json.loads(path.read_text(encoding="utf-8"))


def assess_file(path: Path, expect_event: bool = True, cfg: dict | None = None) -> QualityResult:
    """Decode a file and assess it. Decoding failures and truncation are encoding problems."""
    cfg = cfg or load_thresholds()
    try:
        declared = sf.info(path).frames
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # any decoder error means the file cannot be used
        return QualityResult("Unusable", [f"encoding problem: cannot decode ({exc.__class__.__name__})"])
    result = assess(audio, sr, expect_event, cfg)
    if declared > 0 and audio.shape[0] < 0.99 * declared:
        missing = 1 - audio.shape[0] / declared
        _worsen(result, UNUSABLE if missing > 0.5 else POOR,
                f"missing audio frames: {missing:.0%} of the declared length could not be read")
        result.metrics["missing_fraction"] = round(missing, 4)
    return result


def assess(audio: np.ndarray, sr: int, expect_event: bool = True, cfg: dict | None = None) -> QualityResult:
    """Assess decoded audio shaped (frames[, channels]).

    ``expect_event=False`` skips the excessive-noise check (for audio that is
    background noise by definition).
    """
    cfg = cfg or load_thresholds()
    audio = audio if audio.ndim == 2 else audio[:, np.newaxis]
    result = QualityResult("Good")

    if not np.isfinite(audio).all():
        return QualityResult("Unusable", ["encoding problem: invalid (NaN/infinite) samples"])
    y = to_mono(audio)
    duration = y.size / sr if sr else 0.0
    result.metrics.update(duration_s=round(duration, 3), sample_rate=sr, channels=audio.shape[1])

    # 1. Duration
    if duration < cfg["min_duration_s"]:
        _worsen(result, UNUSABLE, f"unsuitable duration: {duration:.2f} s is shorter than {cfg['min_duration_s']} s")
        return result
    if duration > cfg["max_duration_s"]:
        _worsen(result, POOR, f"unsuitable duration: {duration / 60:.1f} min is longer than {cfg['max_duration_s'] / 60:.0f} min")

    # 2. Silence
    peak_db = _db(np.abs(audio).max())
    frame_db = _frame_levels(y, sr)
    silent = frame_db < cfg["silence_peak_dbfs"]
    result.metrics.update(peak_dbfs=round(peak_db, 1), silent_fraction=round(float(silent.mean()), 3))
    if peak_db < cfg["silence_peak_dbfs"]:
        _worsen(result, UNUSABLE, f"silence: peak {peak_db:.0f} dBFS is below {cfg['silence_peak_dbfs']} dBFS")
        return result
    if silent.mean() > cfg["mostly_silent_fraction"]:
        _worsen(result, POOR, f"silence: {silent.mean():.0%} of the recording is silent")

    # 3. Low signal strength (level of the active part only)
    active_db = _db(np.sqrt(np.mean(10 ** (frame_db[~silent] / 10)))) if (~silent).any() else peak_db
    result.metrics["active_level_dbfs"] = round(active_db, 1)
    low = cfg["low_signal_dbfs"]
    if active_db < low["poor"]:
        _worsen(result, POOR, f"low signal strength: active level {active_db:.0f} dBFS")
    elif active_db < low["acceptable"]:
        _worsen(result, ACCEPTABLE, f"low signal strength: active level {active_db:.0f} dBFS")

    # 4. Clipping (runs at full scale in any channel)
    clipped = _clipped_fraction(audio, cfg["clip_level"], cfg["clip_run_samples"])
    result.metrics["clipped_fraction"] = round(clipped, 5)
    cf = cfg["clipped_fraction"]
    if clipped >= cf["poor"]:
        _worsen(result, POOR, f"clipping: {clipped:.1%} of samples are clipped")
    elif clipped >= cf["acceptable"]:
        _worsen(result, ACCEPTABLE, f"clipping: {clipped:.2%} of samples are clipped")

    # 5. Background noise estimate (FR xiv) and excessive noise
    audible = frame_db[~silent] if (~silent).any() else frame_db
    noise_floor = float(np.percentile(audible, 10))
    dynamic_range = float(np.percentile(audible, 95) - noise_floor)
    flatness = _median_flatness(y)
    result.metrics.update(noise_floor_dbfs=round(noise_floor, 1), dynamic_range_db=round(dynamic_range, 1),
                          spectral_flatness=round(flatness, 3))
    if expect_event:
        nz = cfg["noise"]
        if dynamic_range < nz["dynamic_range_db"]["poor"] and flatness > nz["flatness"]["poor"]:
            _worsen(result, POOR, f"excessive noise: steady noise-like signal (range {dynamic_range:.0f} dB)")
        elif dynamic_range < nz["dynamic_range_db"]["acceptable"] and flatness > nz["flatness"]["acceptable"]:
            _worsen(result, ACCEPTABLE, f"excessive noise: little contrast above the noise (range {dynamic_range:.0f} dB)")

    # 6. Missing frames: interior runs of exact zeros
    count, total = _dropouts(y, sr, cfg["dropout_ms"])
    result.metrics["dropouts"] = count
    if count >= cfg["dropouts"]["poor"]:
        _worsen(result, POOR, f"missing audio frames: {count} dropouts ({total:.2f} s)")
    elif count >= cfg["dropouts"]["acceptable"]:
        _worsen(result, ACCEPTABLE, f"missing audio frames: {count} dropout{'s' if count > 1 else ''} ({total:.2f} s)")

    # 7. Sample rate
    msr = cfg["min_sample_rate"]
    if sr < msr["poor"]:
        _worsen(result, POOR, f"low sample rate: {sr} Hz")
    elif sr < msr["acceptable"]:
        _worsen(result, ACCEPTABLE, f"low sample rate: {sr} Hz (content above {sr // 2} Hz is missing)")
    return result


def _worsen(result: QualityResult, level: int, issue: str) -> None:
    result.issues.append(issue)
    if level > GRADES.index(result.grade):
        result.grade = GRADES[level]


def _db(x: float) -> float:
    return float(20 * np.log10(x + EPS))


def _frame_levels(y: np.ndarray, sr: int, frame_s: float = 0.025) -> np.ndarray:
    """RMS level (dBFS) of consecutive 25 ms frames."""
    n = max(1, int(frame_s * sr))
    usable = y[: (y.size // n) * n] if y.size >= n else np.pad(y, (0, n - y.size))
    frames = usable.reshape(-1, n).astype(np.float64)
    return 10 * np.log10(np.mean(frames**2, axis=1) + EPS)


def _clipped_fraction(audio: np.ndarray, level: float, run: int) -> float:
    at_max = np.abs(audio) >= level
    clipped = 0
    for ch in range(audio.shape[1]):
        mask = at_max[:, ch].astype(np.int8)
        if not mask.any():
            continue
        # Lengths of runs of consecutive True values
        edges = np.diff(np.concatenate([[0], mask, [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        lengths = ends - starts
        clipped += int(lengths[lengths >= run].sum())
    return clipped / audio.size


def _median_flatness(y: np.ndarray, n_fft: int = 1024) -> float:
    """Median spectral flatness over frames: 1 for white noise, near 0 for a tone."""
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[:: n_fft // 2]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2 + EPS
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
    loud = np.sum(spectrum, axis=1) > np.percentile(np.sum(spectrum, axis=1), 20)  # ignore near-silent frames
    return float(np.median(flatness[loud] if loud.any() else flatness))


def _dropouts(y: np.ndarray, sr: int, min_ms: float) -> tuple[int, float]:
    """Interior runs of exact zeros longer than ``min_ms`` (count, total seconds)."""
    zero = (y == 0).astype(np.int8)
    if not zero.any():
        return 0, 0.0
    edges = np.diff(np.concatenate([[0], zero, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    min_len = int(min_ms / 1000 * sr)
    interior = [(s, e) for s, e in zip(starts, ends) if s > 0 and e < y.size and e - s >= min_len]
    return len(interior), sum(e - s for s, e in interior) / sr
