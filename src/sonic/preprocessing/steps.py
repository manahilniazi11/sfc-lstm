"""Individual preprocessing steps. Each takes and returns a float32 numpy array.

Kept as small pure functions so each one can be tested and explained alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

EPS = 1e-10


def load(path: Path) -> tuple[np.ndarray, int]:
    """Decode a file to float32 samples shaped (frames, channels)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(frames, channels) -> (frames,).

    Stereo is averaged. Recordings with more than two channels are microphone
    arrays (MIMII uses 8 mics); averaging them acts like a beamformer and
    changes the sound, so the first microphone is used, like a single mic.
    """
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.shape[1] > 2:
        return audio[:, 0].astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """High-quality resampling (soxr). Downsampling also low-passes at target_sr / 2."""
    if sr == target_sr:
        return y
    return soxr.resample(y, sr, target_sr, quality="HQ").astype(np.float32)


def remove_dc(y: np.ndarray) -> np.ndarray:
    """Remove a constant offset some recorders add; it distorts energy measures."""
    return (y - y.mean()).astype(np.float32) if y.size else y


def reduce_noise(y: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Spectral gating: estimate the noise spectrum and attenuate it by ``amount`` (0-1)."""
    import noisereduce  # imported lazily: only needed when the step is enabled

    return noisereduce.reduce_noise(y=y, sr=sr, prop_decrease=amount, stationary=True).astype(np.float32)


def trim_bounds(y: np.ndarray, top_db: float, frame: int = 512, hop: int = 128) -> tuple[int, int]:
    """Sample range left after cutting leading/trailing audio ``top_db`` below the loudest frame."""
    if y.size < frame:
        return 0, y.size
    rms = frame_rms_db(y, frame, hop)
    loud = np.flatnonzero(rms > rms.max() - top_db)
    if loud.size == 0:
        return 0, 0
    return int(loud[0] * hop), int(min(y.size, loud[-1] * hop + frame))


def frame_rms_db(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """RMS level in dBFS of each (frame, hop) window."""
    if y.size < frame:
        return np.array([rms_dbfs(y)])
    windows = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    return 10 * np.log10(np.mean(windows.astype(np.float64) ** 2, axis=1) + EPS)


def rms_dbfs(y: np.ndarray) -> float:
    if y.size == 0:
        return -np.inf
    return float(10 * np.log10(np.mean(y.astype(np.float64) ** 2) + EPS))


def add_noise_floor(y: np.ndarray, noise_dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """Add faint white noise so a segment never contains exact digital silence.

    Edited clips often contain stretches of exact zeros (a real microphone
    never outputs them); some datasets have many, others none, which lets a
    model identify the source dataset instead of the sound. Applied after
    normalization, so every segment gets the same absolute noise floor.
    """
    noise = rng.standard_normal(y.size).astype(np.float32) * np.float32(10 ** (noise_dbfs / 20))
    return (y + noise).astype(np.float32)


def place(y: np.ndarray, length: int, rng: np.random.Generator, centred: bool = False) -> np.ndarray:
    """Put a clip shorter than ``length`` into a ``length``-sample buffer.

    Training uses a random offset (an event can fall anywhere in a live
    window); inference centres it so results are repeatable. The rest is
    filled by the noise floor added afterwards, never left as digital zeros.
    """
    if y.size >= length:
        return y[:length]
    out = np.zeros(length, dtype=np.float32)
    spare = length - y.size
    offset = spare // 2 if centred else int(rng.integers(0, spare + 1))
    out[offset : offset + y.size] = y
    return out


def normalize(y: np.ndarray, target_rms_dbfs: float, peak_limit_dbfs: float, silence_dbfs: float) -> np.ndarray:
    """Scale to a target RMS level, without letting peaks exceed the limit.

    Silent input is returned unchanged: amplifying it would only amplify noise.
    """
    level = rms_dbfs(y)
    if level < silence_dbfs:
        return y
    gain = 10 ** ((target_rms_dbfs - level) / 20)
    peak = float(np.abs(y).max()) * gain
    peak_limit = 10 ** (peak_limit_dbfs / 20)
    if peak > peak_limit:
        gain *= peak_limit / peak
    return (y * gain).astype(np.float32)
