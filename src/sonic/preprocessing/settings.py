"""Preprocessing settings loaded from ``config/audio.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from ..dataset.config import CONFIG_DIR, ConfigError


@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int = 16000
    segment_seconds: float = 1.0
    hop_seconds: float = 0.5
    trim_top_db: float = 40.0
    target_rms_dbfs: float = -20.0
    peak_limit_dbfs: float = -1.0
    silence_dbfs: float = -60.0
    pad_noise_dbfs: float = -70.0
    event_window_db: float = 20.0
    max_segments_per_clip: int = 5
    noise_reduction: bool = False
    noise_reduction_amount: float = 0.6

    @property
    def segment_samples(self) -> int:
        return round(self.segment_seconds * self.sample_rate)

    @property
    def hop_samples(self) -> int:
        return round(self.hop_seconds * self.sample_rate)


def load_settings(path: Path | None = None) -> AudioSettings:
    path = path or CONFIG_DIR / "audio.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc

    values = {k: v for k, v in data.items() if not k.startswith("_") and k != "noise_reduction"}
    nr = data.get("noise_reduction", {})
    values["noise_reduction"] = bool(nr.get("enabled", False))
    values["noise_reduction_amount"] = float(nr.get("prop_decrease", 0.6))

    known = {f.name for f in fields(AudioSettings)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"{path.name}: unknown setting(s) {sorted(unknown)}")
    settings = AudioSettings(**values)
    if settings.hop_samples <= 0 or settings.hop_seconds > settings.segment_seconds:
        raise ConfigError(f"{path.name}: hop_seconds must be > 0 and <= segment_seconds")
    return settings
