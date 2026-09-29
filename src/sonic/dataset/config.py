"""Loading and validating the class list and the source label map."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# src/sonic/dataset/config.py -> repository root is three levels above "sonic".
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = _PACKAGE_ROOT if (_PACKAGE_ROOT / "config").is_dir() else Path.cwd()


def read_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from a .env file (no quoting rules needed here)."""
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name: str, default: str = "") -> str:
    """An environment variable wins over the repo's .env file, which wins over the default."""
    return os.environ.get(name) or read_env_file(REPO_ROOT / ".env").get(name) or default


CONFIG_DIR = REPO_ROOT / "config"
# Audio is large, so it can live on another drive: set SONIC_DATA_DIR in .env.
# Metadata stays in the repository so it can be committed.
AUDIO_DATA_DIR = Path(setting("SONIC_DATA_DIR", str(REPO_ROOT / "data")))
DEFAULT_RAW_DIR = AUDIO_DATA_DIR / "raw"
DEFAULT_OUT_DIR = AUDIO_DATA_DIR / "audio_dataset"
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "metadata"


class ConfigError(ValueError):
    """Raised when a config file is missing or inconsistent."""


@dataclass(frozen=True)
class SoundClass:
    name: str
    slug: str
    code: str
    mandatory: bool
    critical: bool = False  # SRS NFR 4: recall must be >= 85 %


def load_classes(path: Path | None = None) -> list[SoundClass]:
    path = path or CONFIG_DIR / "sound_classes.json"
    data = _read_json(path)
    classes = [
        SoundClass(c["name"], c["slug"], c["code"], bool(c.get("mandatory", False)), bool(c.get("critical", False)))
        for c in data["classes"]
    ]
    for field in ("name", "slug", "code"):
        values = [getattr(c, field) for c in classes]
        duplicates = {v for v in values if values.count(v) > 1}
        if duplicates:
            raise ConfigError(f"{path.name}: duplicate class {field}(s): {sorted(duplicates)}")
    return classes


def load_label_map(
    classes: list[SoundClass], path: Path | None = None
) -> dict[str, dict[str, str | None]]:
    """Return ``{source: {source_label: class_name or None}}``.

    Every class name used in the map must exist in the class list, so a typo
    fails loudly instead of silently creating an extra class. Keys starting
    with ``_`` are adapter options (e.g. ``_exclude_if_present``), not labels.
    """
    path = path or CONFIG_DIR / "dataset_label_map.json"
    data = _read_json(path)
    known = {c.name for c in classes}
    label_map: dict[str, dict[str, str | None]] = {}
    for source, mapping in data.items():
        if source.startswith("_"):
            continue
        labels = {k: v for k, v in mapping.items() if not k.startswith("_")}
        unknown = {v for v in labels.values() if v is not None and v not in known}
        if unknown:
            raise ConfigError(
                f"{path.name}: source '{source}' maps to unknown class(es) {sorted(unknown)}"
            )
        label_map[source] = mapping
    return label_map


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
