"""Saved models with versions (FR lxxv: every prediction is linked to a model version).

Each model lives in ``python_models/<version>/``: the model file(s) and a
``meta.json`` holding the class list, feature fingerprint, hyperparameters,
calibration temperature, clip aggregation and validation metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..dataset.config import REPO_ROOT

MODELS_DIR = REPO_ROOT / "python_models"


def new_version(name: str, models_dir: Path = MODELS_DIR) -> str:
    """``<name>-<YYYYMMDD>-<n>``, e.g. ``svm-20260925-1``."""
    day = datetime.now().strftime("%Y%m%d")
    n = 1
    while (models_dir / f"{name}-{day}-{n}").exists():
        n += 1
    return f"{name}-{day}-{n}"


def save_meta(model_dir: Path, meta: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_meta(model_dir: Path) -> dict:
    return json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))


def latest(name: str, models_dir: Path = MODELS_DIR) -> Path | None:
    versions = sorted((p for p in models_dir.glob(f"{name}-*") if (p / "meta.json").exists()),
                      key=lambda p: (p.name.split("-")[1], int(p.name.split("-")[2])))
    return versions[-1] if versions else None
