"""Decoding uploaded audio and reading its metadata (FR iv, viii, x; SRS Step 3).

WAV, FLAC, OGG and MP3 are decoded by libsndfile (soundfile). M4A (AAC) is
not supported by libsndfile, so it is converted to WAV with FFmpeg first.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_FORMATS = {".wav": "WAV", ".mp3": "MP3", ".flac": "FLAC", ".ogg": "OGG", ".m4a": "M4A"}
NEEDS_FFMPEG = {".m4a"}
# libsndfile subtypes that tell the bit depth; compressed formats have none.
BIT_DEPTHS = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}


class AudioDecodeError(Exception):
    """The file could not be decoded: damaged, truncated, or not really audio."""


@dataclass
class AudioInfo:
    format: str
    duration_s: float
    sample_rate: int
    channels: int
    bit_depth: int | None
    frames: int


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _ffmpeg_to_wav(path: Path, target: Path) -> None:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise AudioDecodeError("M4A needs FFmpeg, which is not installed on the server")
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-vn", "-c:a", "pcm_s16le", str(target)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not target.exists():
        raise AudioDecodeError(f"FFmpeg could not decode the file: {result.stderr.strip()[:200]}")


def decode(path: Path) -> tuple[np.ndarray, int, AudioInfo]:
    """Decode to float32 (frames, channels) and describe the file. Raises AudioDecodeError."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise AudioDecodeError(f"unsupported format {suffix or '(none)'}")
    try:
        if suffix in NEEDS_FFMPEG:
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "decoded.wav"
                _ffmpeg_to_wav(path, wav)
                audio, sr = sf.read(wav, dtype="float32", always_2d=True)
            bit_depth = None  # AAC is compressed; the decoded 16-bit is not the source depth
            declared = audio.shape[0]
        else:
            meta = sf.info(path)
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            bit_depth = BIT_DEPTHS.get(meta.subtype)
            declared = meta.frames
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"the file is damaged or not valid {SUPPORTED_FORMATS[suffix]} audio ({exc.__class__.__name__})") from exc
    if audio.shape[0] == 0:
        raise AudioDecodeError("the file contains no audio frames")
    info = AudioInfo(
        format=SUPPORTED_FORMATS[suffix],
        duration_s=round(audio.shape[0] / sr, 3),
        sample_rate=int(sr),
        channels=int(audio.shape[1]),
        bit_depth=bit_depth,
        frames=int(declared),
    )
    return audio, int(sr), info
