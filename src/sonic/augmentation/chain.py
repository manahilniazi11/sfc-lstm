"""Build one augmented copy: a random chain of transforms, always including background noise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..dataset.config import CONFIG_DIR
from ..preprocessing.settings import AudioSettings
from ..preprocessing.steps import add_noise_floor, normalize, rms_dbfs
from . import transforms as T

# Transforms run in this physical order: the sound is changed at the source,
# travels through the room, is mixed with the environment, then recorded.
ORDER = ["time_shift", "pitch_shift", "time_stretch", "reverb", "distance", "gain", "background", "device"]
OPTIONAL = [name for name in ORDER if name != "background"]


def load_config(path: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "augmentation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def augment(
    y: np.ndarray,
    noise: np.ndarray,
    cfg: dict,
    settings: AudioSettings,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Return the augmented segment and a readable description of what was applied."""
    sr = settings.sample_rate
    low, high = cfg["transforms_per_copy"]
    count = int(rng.integers(low, high + 1))
    chosen = {"background", *rng.choice(OPTIONAL, size=count - 1, replace=False)}
    reference = rms_dbfs(y)  # background SNR is set against the untouched level
    log = []

    for name in ORDER:
        if name not in chosen:
            continue
        if name == "time_shift":
            k = int(rng.choice([-1, 1]) * rng.uniform(0.05, 1) * cfg["time_shift_seconds"] * sr)
            y = T.time_shift(y, k)
            log.append(f"shift({k / sr:+.2f}s)")
        elif name == "pitch_shift":
            n = float(rng.uniform(-1, 1) * cfg["pitch_shift_semitones"])
            y = T.pitch_shift(y, sr, n)
            log.append(f"pitch({n:+.1f}st)")
        elif name == "time_stretch":
            rate = float(rng.uniform(*cfg["time_stretch_rate"]))
            y = T.time_stretch(y, rate)
            log.append(f"stretch({rate:.2f}x)")
        elif name == "reverb":
            rt60 = float(rng.uniform(*cfg["reverb"]["rt60_seconds"]))
            wet = float(rng.uniform(*cfg["reverb"]["wet"]))
            y = T.reverb(y, sr, rt60, wet, rng)
            log.append(f"reverb(rt60={rt60:.2f}s,wet={wet:.2f})")
        elif name == "distance":
            cutoff = float(rng.uniform(*cfg["distance"]["lowpass_hz"]))
            wet = float(rng.uniform(*cfg["distance"]["extra_reverb_wet"]))
            y = T.distance(y, sr, cutoff, wet, rng)
            log.append(f"distance(lowpass={cutoff:.0f}Hz)")
        elif name == "gain":
            db = float(rng.uniform(*cfg["gain_db"]))
            y = T.gain(y, db)
            log.append(f"gain({db:+.1f}dB)")
        elif name == "background":
            snr = float(rng.uniform(*cfg["background_noise"]["snr_db"]))
            y = T.mix_background(y, noise, snr, reference)
            log.append(f"noise(snr={snr:.1f}dB)")
        elif name == "device":
            bands = cfg["device"]["bandpass_hz"]
            band = bands[int(rng.integers(len(bands)))]
            clip = float(rng.uniform(*cfg["device"]["clip_level"]))
            y = T.device(y, sr, tuple(band), clip)
            log.append(f"device({band[0]}-{band[1]}Hz,clip={clip:.2f})")

    # Finish exactly like preprocessing, so augmented segments look like real ones.
    y = normalize(y, settings.target_rms_dbfs, settings.peak_limit_dbfs, settings.silence_dbfs)
    y = add_noise_floor(y, settings.pad_noise_dbfs, rng)
    return y, "+".join(log)
