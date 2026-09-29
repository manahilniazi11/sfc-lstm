"""Train one model with the shared protocol, so the four models are compared fairly.

1. The model's own ``train`` fits on the training split (original + augmented
   segments) and tunes its hyperparameters on validation segments.
2. A calibration temperature is fitted on validation segments, so the
   confidence scores the alert rules rely on mean what they say.
3. Validation clips are cut the way the web app cuts them (every window);
   window scores are combined per clip with mean / max / top-3, and the
   method with the best validation macro F1 is kept.
4. The model, its metadata and a validation confusion matrix are saved.

The test split is not touched here; ``compare`` evaluates all models on it once.
"""

from __future__ import annotations

import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np

from ..dataset.config import REPO_ROOT
from .data import SegmentData, critical_classes, fingerprint, load_segments
from .evaluate import AGGREGATIONS, aggregate, calibrate, compute_metrics, fit_temperature, format_metrics, plot_confusion
from .registry import MODELS_DIR, load_meta, new_version, save_meta
from .windows import load_windows

REPORTS_DIR = REPO_ROOT / "reports"
MODEL_MODULES = {"svm": "svm", "xgboost": "xgb", "cnn": "cnn", "yamnet": "yamnet", "ensemble": "ensemble"}


def model_module(name: str):
    return import_module(f".{MODEL_MODULES[name]}", __package__)


def segment_batch(data: SegmentData, rows: np.ndarray) -> dict:
    return {"X": data.X[rows], "logmel": np.asarray(data.logmel[rows]), "rows": rows, "data": data}


def validate(model, data: SegmentData, critical: list[str]) -> dict:
    """Calibrate and choose the clip aggregation on validation; returns everything saved in meta.json.

    Two temperatures: one for single 1 s windows (live monitoring) and one for
    whole-clip decisions (uploads). Averaging windows makes clip scores less
    extreme than window scores, so one temperature cannot fit both.
    """
    val_rows = data.rows("validation", augmented=False)
    seg_probs = model.predict_proba(segment_batch(data, val_rows))
    temperature = fit_temperature(seg_probs, data.y[val_rows])
    seg_metrics = compute_metrics(data.y[val_rows], calibrate(seg_probs, temperature), data.classes, critical)

    windows = load_windows("validation", data.classes)
    t0 = time.perf_counter()
    win_probs = calibrate(model.predict_proba(windows.batch()), temperature)
    per_window_ms = 1000 * (time.perf_counter() - t0) / max(len(windows.X), 1)
    clip_probs = {m: aggregate(win_probs, windows.offsets, m) for m in AGGREGATIONS}
    clip_f1 = {m: compute_metrics(windows.labels, p, data.classes, critical)["macro_f1"] for m, p in clip_probs.items()}
    best_agg = max(clip_f1, key=clip_f1.get)
    clip_temperature = fit_temperature(clip_probs[best_agg], windows.labels)
    clip_metrics = compute_metrics(windows.labels, calibrate(clip_probs[best_agg], clip_temperature), data.classes, critical)
    return {
        "temperature": round(temperature, 4),
        "clip_temperature": round(clip_temperature, 4),
        "aggregation": best_agg,
        "aggregation_macro_f1": clip_f1,
        "validation": {"segment": seg_metrics, "clip": clip_metrics},
        "inference_ms_per_window": round(per_window_ms, 2),
    }


def report(name: str, version: str, params: dict, v: dict) -> None:
    agg_scores = ", ".join(f"{m} {f:.3f}" for m, f in v["aggregation_macro_f1"].items())
    print(f"\n[{name}] {version}")
    print(f"  parameters: {params}")
    print(f"  temperature: window {v['temperature']:.2f}, clip {v['clip_temperature']:.2f}; "
          f"clip aggregation: {v['aggregation']} ({agg_scores})")
    print(f"  inference: {v['inference_ms_per_window']:.1f} ms per 1 s window\n")
    print(format_metrics(v["validation"]["segment"], "Validation, per 1 s segment"))
    print()
    print(format_metrics(v["validation"]["clip"], "Validation, per clip (as in the app)"))


def _save_reports(version: str, classes: list[str], v: dict) -> None:
    plot_confusion(v["validation"]["clip"]["confusion_matrix"], classes,
                   REPORTS_DIR / "confusion" / f"{version}_validation.png",
                   f"{version}: validation clips ({v['aggregation']} of windows)")


def run(name: str) -> dict:
    module = model_module(name)
    started = time.perf_counter()
    data = load_segments()
    critical = critical_classes()
    train_rows = data.rows("train")
    val_rows = data.rows("validation", augmented=False)
    print(f"[{name}] {len(data.classes)} classes, {len(train_rows)} training / {len(val_rows)} validation segments")

    model, params, tuning = module.train(data, train_rows, val_rows)
    v = validate(model, data, critical)

    version = new_version(name)
    model_dir = MODELS_DIR / version
    model.save(model_dir)
    save_meta(model_dir, {
        "name": name,
        "version": version,
        "classes": data.classes,
        "critical_classes": critical,
        "fingerprint": fingerprint(),
        "params": params,
        **v,
        "train_segments": int(len(train_rows)),
        "train_seconds": round(time.perf_counter() - started, 1),
    })
    (REPORTS_DIR / "tuning").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "tuning" / f"{version}.json").write_text(json.dumps(tuning, indent=1, default=str), encoding="utf-8")
    _save_reports(version, data.classes, v)
    report(name, version, params, v)
    return {"version": version, **v}


def recalibrate(version: str) -> dict:
    """Redo calibration and aggregation choice for a saved model without retraining it."""
    model_dir = MODELS_DIR / version
    meta = load_meta(model_dir)
    if meta["fingerprint"] != fingerprint():
        raise ValueError(f"{version} was trained on different features; retrain it")
    model = model_module(meta["name"]).load(model_dir)
    data = load_segments()
    v = validate(model, data, meta["critical_classes"])
    save_meta(model_dir, {**meta, **v})
    _save_reports(version, data.classes, v)
    report(meta["name"], version, meta["params"], v)
    return v
