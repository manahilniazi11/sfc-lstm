"""Waveform and mel-spectrogram images (FR xxi-xxii).

Matplotlib's object API (Figure + Agg canvas, no pyplot) is used because it
is safe to call from several web requests at once.
"""

from __future__ import annotations

import io

import librosa
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..preprocessing import steps

PLOT_SR = 16000  # the rate the models use; 0-8 kHz is the band they see
MAX_POINTS = 4000  # waveform points drawn (a min/max envelope beyond that); the image is 1000 px wide


def _png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buffer)
    return buffer.getvalue()


def _mono16k(audio: np.ndarray, sr: int) -> np.ndarray:
    return steps.resample(steps.to_mono(audio), sr, PLOT_SR)


def waveform_png(audio: np.ndarray, sr: int, marks: list[tuple[float, float, str]] | None = None) -> bytes:
    """Amplitude over time; `marks` shades (start, end, label) spans, e.g. windows with a detected event."""
    y = _mono16k(audio, sr)
    t = np.arange(y.size) / PLOT_SR
    fig = Figure(figsize=(10, 2.6), dpi=100)
    ax = fig.add_subplot()
    if y.size > MAX_POINTS:  # draw the min/max envelope so short peaks are not lost
        step = int(np.ceil(y.size / (MAX_POINTS // 2)))
        n = y.size // step
        blocks = y[: n * step].reshape(n, step)
        tb = t[: n * step : step]
        ax.fill_between(tb, blocks.min(1), blocks.max(1), color="#2f80ed", linewidth=0)
    else:
        ax.plot(t, y, color="#2f80ed", linewidth=0.6)
    for start, end, label in marks or []:
        ax.axvspan(start, end, color="#dc3545", alpha=0.12)
        ax.text((start + end) / 2, 0.92, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="#9b1c2a")
    peak = max(float(np.abs(y).max()) if y.size else 0.0, 1e-3)
    ax.set_ylim(-peak * 1.05, peak * 1.05)
    ax.set_xlim(0, max(t[-1] if t.size else 0.0, 0.01))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def spectrogram_png(audio: np.ndarray, sr: int) -> bytes:
    """Mel spectrogram in dB, 64 mel bands up to 8 kHz (the same view as the CNN's input)."""
    y = _mono16k(audio, sr)
    mel = librosa.feature.melspectrogram(y=y, sr=PLOT_SR, n_fft=1024, hop_length=256, n_mels=64, fmax=PLOT_SR / 2)
    db = librosa.power_to_db(mel, ref=np.max if mel.size and mel.max() > 0 else 1.0, top_db=80)
    fig = Figure(figsize=(10, 3.2), dpi=100)
    ax = fig.add_subplot()
    duration = y.size / PLOT_SR
    image = ax.imshow(db, origin="lower", aspect="auto", cmap="magma", extent=[0, duration, 0, 64])
    freqs = librosa.mel_frequencies(n_mels=64, fmax=PLOT_SR / 2)
    ticks = [0, 16, 32, 48, 63]
    ax.set_yticks(ticks, [f"{freqs[i] / 1000:.1f}" for i in ticks])
    ax.set_ylabel("Frequency (kHz, mel scale)")
    ax.set_xlabel("Time (s)")
    fig.colorbar(image, ax=ax, format="%+.0f dB", pad=0.01)
    fig.tight_layout()
    return _png(fig)
