"""Augmentation transforms (SRS Hint: noise, shift, pitch, stretch, volume, reverb, distance, device).

Every function takes a 1-D float32 segment and returns a new one of the same
length. Randomness comes only from the ``rng`` passed in, so a copy can be
regenerated exactly from its seed.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from ..preprocessing.steps import rms_dbfs

EPS = 1e-10


def fit_length(y: np.ndarray, length: int) -> np.ndarray:
    """Centre-crop or zero-pad to exactly ``length`` samples."""
    if y.size > length:
        start = (y.size - length) // 2
        return y[start : start + length]
    out = np.zeros(length, dtype=np.float32)
    start = (length - y.size) // 2
    out[start : start + y.size] = y
    return out


def mix_background(y: np.ndarray, noise: np.ndarray, snr_db: float, reference_dbfs: float) -> np.ndarray:
    """Add ``noise`` so the signal-to-noise ratio is ``snr_db`` relative to ``reference_dbfs``.

    The reference is the segment's level *before* gain/distance changes, so a
    source made quieter by those transforms really ends up closer to the noise,
    like a distant sound in the same room.
    """
    noise = fit_length(np.resize(noise, max(noise.size, y.size)), y.size)
    noise_level = rms_dbfs(noise)
    if not np.isfinite(noise_level):
        return y
    target = reference_dbfs - snr_db
    return (y + noise * np.float32(10 ** ((target - noise_level) / 20))).astype(np.float32)


def time_shift(y: np.ndarray, k: int) -> np.ndarray:
    """Move the content by ``k`` samples (negative = earlier); the gap is filled by the noise floor later."""
    out = np.zeros_like(y)
    if k >= 0:
        out[k:] = y[: y.size - k]
    else:
        out[:k] = y[-k:]
    return out


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Faster (rate > 1) or slower playback without changing pitch, kept at the same length."""
    return fit_length(librosa.effects.time_stretch(y, rate=rate).astype(np.float32), y.size)


def gain(y: np.ndarray, db: float) -> np.ndarray:
    """Volume adjustment. Matters because noise is mixed relative to the original level."""
    return (y * np.float32(10 ** (db / 20))).astype(np.float32)


def room_impulse(sr: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic room response: exponentially decaying noise that falls 60 dB in ``rt60`` seconds."""
    t = np.arange(int(rt60 * sr)) / sr
    ir = rng.standard_normal(t.size) * np.exp(-6.908 * t / rt60)  # ln(1000) = 6.908 -> -60 dB
    ir[0] = 1.0  # direct sound
    return (ir / np.sqrt(np.sum(ir**2))).astype(np.float32)


def reverb(y: np.ndarray, sr: int, rt60: float, wet: float, rng: np.random.Generator) -> np.ndarray:
    """Blend in a reverberant copy (``wet`` 0-1), keeping the overall level."""
    wet_signal = fftconvolve(y, room_impulse(sr, rt60, rng))[: y.size]
    out = (1 - wet) * y + wet * wet_signal
    scale = np.sqrt((np.mean(y**2) + EPS) / (np.mean(out**2) + EPS))
    return (out * scale).astype(np.float32)


def distance(y: np.ndarray, sr: int, lowpass_hz: float, extra_wet: float, rng: np.random.Generator) -> np.ndarray:
    """A far-away source: air absorbs high frequencies, more reverb, lower level."""
    sos = butter(4, lowpass_hz, btype="lowpass", fs=sr, output="sos")
    far = sosfilt(sos, y).astype(np.float32)
    far = reverb(far, sr, rt60=float(rng.uniform(0.3, 0.9)), wet=extra_wet, rng=rng)
    return gain(far, float(rng.uniform(-8, -3)))


def device(y: np.ndarray, sr: int, band_hz: tuple[float, float], clip_level: float) -> np.ndarray:
    """A cheap microphone: limited frequency band and soft clipping when overdriven."""
    low, high = band_hz
    sos = butter(4, [low, min(high, sr / 2 - 100)], btype="bandpass", fs=sr, output="sos")
    band = sosfilt(sos, y).astype(np.float32)
    peak = float(np.abs(band).max()) + EPS
    driven = band / peak  # full scale, then soft-clip above clip_level
    clipped = np.tanh(driven / clip_level) * clip_level
    return (clipped * peak).astype(np.float32)
