"""The preprocessing pipeline: raw audio in, fixed-length normalized segments out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import steps
from .settings import AudioSettings, load_settings

TRAINING, INFERENCE = "training", "inference"


@dataclass
class Segment:
    """One fixed-length window, ready for feature extraction or the GTM model."""

    start_s: float  # position in the original recording
    end_s: float
    samples: np.ndarray  # float32, mono, settings.sample_rate, exactly segment_samples long
    level_dbfs: float  # RMS level before normalization
    silent: bool  # below settings.silence_dbfs: do not classify
    padded: bool  # the recording was shorter than one segment


def preprocess(
    audio: np.ndarray,
    sr: int,
    settings: AudioSettings | None = None,
    mode: str = INFERENCE,
    seed: int = 0,
) -> list[Segment]:
    """Turn raw audio (frames[, channels]) into segments.

    ``inference`` returns every window in time order, so an uploaded clip or
    live stream is analysed end to end (silent windows are flagged).
    ``training`` keeps only windows that contain the event: within
    ``event_window_db`` of the loudest window, non-overlapping, at most
    ``max_segments_per_clip``. A 0.5 s gunshot in a 4 s clip therefore yields
    the window with the shot, not three windows of background labelled Gunshot.
    """
    if mode not in (TRAINING, INFERENCE):
        raise ValueError(f"mode must be '{TRAINING}' or '{INFERENCE}'")
    s = settings or load_settings()
    rng = np.random.default_rng(seed)

    y = steps.to_mono(audio)
    y = steps.resample(y, sr, s.sample_rate)
    y = steps.remove_dc(y)
    if s.noise_reduction and y.size:
        y = steps.reduce_noise(y, s.sample_rate, s.noise_reduction_amount)
    start, end = steps.trim_bounds(y, s.trim_top_db)
    y = y[start:end]
    if y.size == 0:
        return []

    windows = _windows(y, s)
    if mode == TRAINING:
        windows = _event_windows(windows, s)

    segments = []
    for offset, content in windows:
        # Level and loudness are measured on the real content only, so a short
        # clip is not judged quieter because of the padding around it.
        level = steps.rms_dbfs(content)
        silent = level < s.silence_dbfs
        if not silent:
            content = steps.normalize(content, s.target_rms_dbfs, s.peak_limit_dbfs, s.silence_dbfs)
        padded = content.size < s.segment_samples
        samples = steps.place(content, s.segment_samples, rng, centred=(mode == INFERENCE))
        samples = steps.add_noise_floor(samples, s.pad_noise_dbfs, rng)
        begin = (start + offset) / s.sample_rate
        segments.append(Segment(
            round(begin, 3), round(begin + content.size / s.sample_rate, 3), samples, round(level, 2), silent, padded
        ))
    return segments


def preprocess_file(path: Path, settings: AudioSettings | None = None, mode: str = INFERENCE, seed: int = 0) -> list[Segment]:
    audio, sr = steps.load(Path(path))
    return preprocess(audio, sr, settings, mode, seed)


def _windows(y: np.ndarray, s: AudioSettings) -> list[tuple[int, np.ndarray]]:
    """(offset, content) for every hop-spaced window; the last one is aligned to the end.

    A recording shorter than one segment is a single, shorter window; it is
    placed into a full-length segment after normalization.
    """
    n, size = y.size, s.segment_samples
    if n <= size:
        return [(0, y.copy())]
    offsets = list(range(0, n - size + 1, s.hop_samples))
    if offsets[-1] != n - size:
        offsets.append(n - size)  # cover the tail
    return [(o, y[o : o + size].copy()) for o in offsets]


def _event_windows(windows: list, s: AudioSettings) -> list:
    """Loudest windows first, skipping windows that overlap one already chosen."""
    if len(windows) == 1:
        return windows
    levels = [steps.rms_dbfs(w[1]) for w in windows]
    loudest = max(levels)
    candidates = sorted(
        (i for i, lvl in enumerate(levels) if lvl >= loudest - s.event_window_db and lvl >= s.silence_dbfs),
        key=lambda i: -levels[i],
    )
    chosen: list[int] = []
    for i in candidates:
        if all(abs(windows[i][0] - windows[j][0]) >= s.segment_samples for j in chosen):
            chosen.append(i)
        if len(chosen) == s.max_segments_per_clip:
            break
    return [windows[i] for i in sorted(chosen)]
