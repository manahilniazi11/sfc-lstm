"""Model 5: ensemble of YAMNet and the CNN (soft voting).

Each member scores a window with its own calibrated probabilities; the
ensemble averages them. The two members make different mistakes (on the
first comparison YAMNet missed Aggression more, the CNN missed Panic
Scream more), so the average is right more often than either alone.
The ensemble then gets its own calibration and clip aggregation, like any
other model, and competes under the same selection rule.

Disclosure for the report: the ensemble was added after the first test-set
comparison showed the two models' complementary errors. It is judged on
validation like every other model, but the idea itself came from test results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MEMBERS = ("yamnet", "cnn")
NAME = "ensemble"


class EnsembleModel:
    def __init__(self, members: list[tuple[object, dict]]):
        self.members = members  # (model, meta) per member

    @property
    def versions(self) -> list[str]:
        return [meta["version"] for _, meta in self.members]

    def predict_proba(self, batch: dict) -> np.ndarray:
        from .evaluate import calibrate

        probs = [calibrate(model.predict_proba(batch), meta["temperature"]) for model, meta in self.members]
        return np.mean(probs, axis=0)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "members.json").write_text(json.dumps({"members": self.versions}, indent=1), encoding="utf-8")


def _load_members(versions: list[str]) -> EnsembleModel:
    from .compare import load_model

    return EnsembleModel([load_model(v) for v in versions])


def load(model_dir: Path) -> EnsembleModel:
    versions = json.loads((model_dir / "members.json").read_text(encoding="utf-8"))["members"]
    return _load_members(versions)


def train(data, train_rows: np.ndarray, val_rows: np.ndarray):
    """Nothing to fit: combine the latest saved member models (they must use the same classes)."""
    from .registry import latest

    paths = [latest(name) for name in MEMBERS]
    if any(p is None for p in paths):
        raise ValueError(f"train {MEMBERS} first")
    model = _load_members([p.name for p in paths])
    for _, meta in model.members:
        if meta["classes"] != data.classes:
            raise ValueError(f"{meta['version']} was trained on different classes; retrain it")
    print(f"  members: {', '.join(model.versions)}")
    return model, {"members": model.versions, "combination": "mean of calibrated window probabilities"}, []
