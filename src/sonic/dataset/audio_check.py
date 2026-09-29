"""Reading audio file properties and rejecting unusable files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}

_BIT_DEPTH = {
    "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24,
    "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64,
}


class AudioCheckError(Exception):
    """The file cannot be used; the message is the rejection reason."""


@dataclass(frozen=True)
class AudioInfo:
    format: str
    duration_s: float
    sample_rate: int
    channels: int
    bit_depth: int | None
    file_size_bytes: int
    peak_amplitude: float
    sha256: str


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> AudioInfo:
    """Decode the whole file once to prove it is intact and measure its peak."""
    if not path.is_file():
        raise AudioCheckError("missing file")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise AudioCheckError(f"unsupported extension {path.suffix}")
    try:
        info = sf.info(path)
        peak = 0.0
        for block in sf.blocks(path, blocksize=65536, dtype="float32"):
            if block.size:
                peak = max(peak, float(np.abs(block).max()))
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        hint = " (convert to WAV with ffmpeg)" if path.suffix.lower() == ".m4a" else ""
        raise AudioCheckError(f"undecodable: {exc}{hint}") from exc

    return AudioInfo(
        format=info.format,
        duration_s=round(info.frames / info.samplerate, 3) if info.samplerate else 0.0,
        sample_rate=info.samplerate,
        channels=info.channels,
        bit_depth=_BIT_DEPTH.get(info.subtype),
        file_size_bytes=path.stat().st_size,
        peak_amplitude=round(peak, 5),
        sha256=sha256_of(path),
    )


def rejection_reason(info: AudioInfo, min_duration_s: float, silence_peak: float) -> str | None:
    """Why a decodable file is still unusable, or None if it is fine."""
    if info.duration_s <= 0:
        return "empty audio"
    if info.duration_s < min_duration_s:
        return f"too short ({info.duration_s:.2f}s < {min_duration_s}s)"
    if info.peak_amplitude < silence_peak:
        return f"silent (peak {info.peak_amplitude:.5f} < {silence_peak})"
    return None
