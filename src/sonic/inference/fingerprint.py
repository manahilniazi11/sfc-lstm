"""Near-duplicate detection (FR lxxiv): re-encoded, trimmed or volume-adjusted copies.

A SHA-256 hash only catches byte-identical files. For near-duplicates each
recording gets a coarse "acoustic fingerprint": the loudness of 16 frequency
bands every 0.25 s, in dB, with the recording's average subtracted.

- volume change: adds the same number of dB everywhere, removed by the subtraction
- re-encoding (e.g. MP3): changes fine detail, not the coarse band energies
- trimming: the shorter fingerprint is slid along the longer one and the best
  alignment counts

Two recordings are near-duplicates when the best alignment correlates above
`THRESHOLD` over at least `MIN_FRAMES` frames.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..preprocessing import steps

SR = 16000
HOP_S = 0.25
N_BANDS = 16
MAX_SECONDS = 120  # longer recordings are compared on their first two minutes
MIN_FRAMES = 4  # at least 1 s must overlap
THRESHOLD = 0.9
SILENCE_DBFS = -60.0


def fingerprint(audio: np.ndarray, sr: int) -> np.ndarray:
    """(frames, 16) float32; empty when the recording is silent."""
    y = steps.resample(steps.to_mono(audio), sr, SR)
    start, end = steps.trim_bounds(y, 40)
    y = y[start:end][: MAX_SECONDS * SR]
    if y.size < SR * HOP_S or steps.rms_dbfs(y) < SILENCE_DBFS:  # nothing to compare
        return np.zeros((0, N_BANDS), np.float32)
    hop = int(SR * HOP_S)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=hop, n_mels=N_BANDS, fmax=SR / 2, center=False)
    db = 10 * np.log10(mel + 1e-10)
    return (db - db.mean()).T.astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best correlation between the two fingerprints over all alignments (0 when too short to judge)."""
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < MIN_FRAMES:
        return 0.0
    s = short - short.mean()
    s_norm = np.linalg.norm(s)
    if s_norm == 0:
        return 0.0
    best = 0.0
    for offset in range(len(long_) - len(short) + 1):
        w = long_[offset : offset + len(short)]
        w = w - w.mean()
        w_norm = np.linalg.norm(w)
        if w_norm:
            best = max(best, float((s * w).sum() / (s_norm * w_norm)))
    return best


def to_bytes(fp: np.ndarray) -> bytes:
    return fp.astype(np.float16).tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(-1, N_BANDS)
