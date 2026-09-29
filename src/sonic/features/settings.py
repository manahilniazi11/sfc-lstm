"""Feature settings loaded from ``config/features.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from ..dataset.config import CONFIG_DIR, ConfigError


@dataclass(frozen=True)
class FeatureSettings:
    n_fft: int = 512
    hop_length: int = 256
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 8000.0
    n_mfcc: int = 20
    mel_summary_bands: int = 16


def load_feature_settings(path: Path | None = None) -> FeatureSettings:
    path = path or CONFIG_DIR / "features.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    values = {k: v for k, v in data.items() if not k.startswith("_")}
    unknown = set(values) - {f.name for f in fields(FeatureSettings)}
    if unknown:
        raise ConfigError(f"{path.name}: unknown setting(s) {sorted(unknown)}")
    settings = FeatureSettings(**values)
    if settings.n_mels % settings.mel_summary_bands:
        raise ConfigError(f"{path.name}: n_mels must be a multiple of mel_summary_bands")
    return settings
