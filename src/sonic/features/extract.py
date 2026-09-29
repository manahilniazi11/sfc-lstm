"""Acoustic features for one preprocessed segment (SRS Step 6, FR xx).

``extract`` returns both representations from a single STFT:

* ``vector``: 123 summary statistics for the classical models (SVM, Random
  Forest, gradient boosting). Names come from ``feature_names``.
* ``log_mel``: a (n_mels, frames) log-mel spectrogram, the CNN's input image.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .settings import FeatureSettings

# Spectral contrast: 6 octave bands above 100 Hz reach 6.4 kHz, below the 8 kHz Nyquist.
CONTRAST_FMIN, CONTRAST_BANDS = 100.0, 6


@dataclass
class Features:
    vector: np.ndarray  # float32, len(feature_names(settings))
    log_mel: np.ndarray  # float32, (n_mels, frames)


def extract(y: np.ndarray, sr: int, s: FeatureSettings) -> Features:
    magnitude = np.abs(librosa.stft(y, n_fft=s.n_fft, hop_length=s.hop_length))
    power = magnitude**2
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=s.n_mels, fmin=s.fmin, fmax=s.fmax)
    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=80.0)

    frames = _frame_features(y, sr, s, magnitude, power, log_mel)
    vector = np.concatenate([_summarise(frames[name], stats) for name, stats, _ in _layout(s)])
    return Features(vector.astype(np.float32), log_mel.astype(np.float32))


def feature_names(s: FeatureSettings) -> list[str]:
    names = []
    for name, stats, rows in _layout(s):
        for stat in stats:
            names.extend(f"{name}_{i}_{stat}" if rows > 1 else f"{name}_{stat}" for i in range(rows))
    return names


def _layout(s: FeatureSettings) -> list[tuple[str, tuple[str, ...], int]]:
    """(feature, statistics, rows) in vector order. Changing this changes the model input."""
    return [
        ("mfcc", ("mean", "std"), s.n_mfcc),           # timbre: the spectral envelope
        ("mfcc_delta", ("std",), s.n_mfcc),            # how fast the timbre changes
        ("chroma", ("mean", "std"), 12),               # energy per pitch class (sirens, horns)
        ("mel_band", ("mean",), s.mel_summary_bands),  # coarse loudness per frequency region
        ("zcr", ("mean", "std"), 1),                   # noisiness / brightness
        ("rms", ("mean", "std", "max"), 1),            # energy and how impulsive it is
        ("centroid", ("mean", "std"), 1),              # spectral "centre of mass" in Hz
        ("bandwidth", ("mean", "std"), 1),             # spread around the centroid
        ("rolloff", ("mean", "std"), 1),               # frequency below which 85 % of energy lies
        ("contrast", ("mean",), CONTRAST_BANDS + 1),   # peaks vs valleys per octave (tonal vs noise)
        ("flatness", ("mean", "std"), 1),              # 1 = white noise, 0 = pure tone
        ("onset", ("mean", "std", "max"), 1),          # sudden energy increases (gunshots, glass)
    ]


def _frame_features(y, sr, s: FeatureSettings, magnitude, power, log_mel) -> dict[str, np.ndarray]:
    """Frame-level features, each shaped (rows, frames)."""
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=s.n_mfcc)
    group = s.n_mels // s.mel_summary_bands
    return {
        "mfcc": mfcc,
        "mfcc_delta": librosa.feature.delta(mfcc),
        # Fixed standard tuning: estimating it per segment fails on unpitched sounds
        # (noise, gunshots) and makes chroma inconsistent between segments.
        "chroma": librosa.feature.chroma_stft(S=power, sr=sr, n_fft=s.n_fft, tuning=0.0),
        "mel_band": log_mel.reshape(s.mel_summary_bands, group, -1).mean(axis=1),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=s.n_fft, hop_length=s.hop_length),
        "rms": librosa.feature.rms(S=magnitude, frame_length=s.n_fft),
        "centroid": librosa.feature.spectral_centroid(S=magnitude, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(S=magnitude, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.85),
        "contrast": librosa.feature.spectral_contrast(S=magnitude, sr=sr, fmin=CONTRAST_FMIN, n_bands=CONTRAST_BANDS),
        "flatness": librosa.feature.spectral_flatness(S=magnitude),
        "onset": librosa.onset.onset_strength(S=log_mel, sr=sr)[np.newaxis, :],
    }


def _summarise(frames: np.ndarray, stats: tuple[str, ...]) -> np.ndarray:
    funcs = {"mean": np.mean, "std": np.std, "max": np.max}
    return np.concatenate([funcs[stat](frames, axis=1) for stat in stats])
