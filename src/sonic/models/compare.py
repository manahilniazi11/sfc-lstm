"""Final comparison of all trained models (SRS Step 7 "best-performing model").

Everything is measured on the **test** split, which no model has seen and
which played no part in tuning, calibration or selection.

Selection rule (fixed before the test set was scored, uses validation only):
1. Eligible: models whose validation clip recall is >= 85 % for every
   critical class present. If no model qualifies, all models stay eligible.
2. Rank by validation clip macro F1.
3. Within 0.01 of the best: prefer higher mean critical recall, then faster
   inference.
Test results, the hidden-test simulator and the factory-shortcut test are
reported for every model; a model that calls more than 20 % of unseen
normal factory recordings "Machinery Fault" is flagged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..dataset.config import DEFAULT_OUT_DIR
from .data import critical_classes, fingerprint, load_segments
from .evaluate import aggregate, calibrate, compute_metrics, format_metrics, plot_confusion
from .registry import MODELS_DIR, latest, load_meta
from .robustness import CONDITIONS, DESCRIPTIONS, condition_windows, factory_windows, load_mono
from .train import REPORTS_DIR, model_module, segment_batch
from .windows import load_windows, window_features

MODEL_ORDER = ["svm", "xgboost", "yamnet", "cnn", "ensemble"]
FACTORY_FLAG = 0.20
MACHINERY = "Machinery Fault"


def load_model(version: str):
    model_dir = MODELS_DIR / version
    meta = load_meta(model_dir)
    if meta["fingerprint"] != fingerprint():
        raise ValueError(f"{version} was trained on different features; retrain it")
    return model_module(meta["name"]).load(model_dir), meta


def clip_probs(model, meta: dict, windows) -> np.ndarray:
    """Window scores -> calibrated -> combined per clip -> calibrated again, exactly as in the app."""
    window = calibrate(model.predict_proba(windows.batch()), meta["temperature"])
    return calibrate(aggregate(window, windows.offsets, meta["aggregation"]), meta["clip_temperature"])


def select(metas: dict[str, dict]) -> tuple[str, str]:
    """Apply the pre-registered rule; returns (version, explanation)."""
    def crit(m):
        return m["validation"]["clip"]["critical_recall"]

    eligible = {v: m for v, m in metas.items() if crit(m) and min(crit(m).values()) >= 0.85}
    note = "all critical classes >= 85 % on validation"
    if not eligible:
        eligible, note = metas, "no model reached 85 % on every critical class, so all were eligible"
    best_f1 = max(m["validation"]["clip"]["macro_f1"] for m in eligible.values())
    close = {v: m for v, m in eligible.items() if m["validation"]["clip"]["macro_f1"] >= best_f1 - 0.01}
    chosen = max(close, key=lambda v: (np.mean(list(crit(close[v]).values())), -close[v]["inference_ms_per_window"]))
    return chosen, note


def upload_latency_ms(model, meta: dict, seconds: float = 30.0) -> float:
    """End-to-end time for a 30 s upload: preprocessing + features + model + combining windows."""
    clip = next((DEFAULT_OUT_DIR / "test").rglob("*.wav"))
    y = load_mono(clip)
    y = np.resize(y, int(seconds * 16000))
    start = time.perf_counter()
    X, logmel, wave = window_features(y, 16000)
    probs = calibrate(model.predict_proba({"X": X, "logmel": logmel, "wave": wave}), meta["temperature"])
    aggregate(probs, np.array([0, len(probs)]), meta["aggregation"])
    return 1000 * (time.perf_counter() - start)


def run(models: list[str] | None = None) -> dict:
    versions = [p.name for n in (models or MODEL_ORDER) if (p := latest(n))]
    loaded = {v: load_model(v) for v in versions}
    data = load_segments()
    classes = data.classes
    critical = [c for c in critical_classes() if c in classes]
    for v, (_, meta) in loaded.items():
        if meta["classes"] != classes:
            raise ValueError(f"{v} was trained on different classes; retrain it")

    test = load_windows("test", classes)
    test_rows = data.rows("test", augmented=False)
    results: dict[str, dict] = {}
    for v, (model, meta) in loaded.items():
        print(f"[{v}] test set")
        probs = clip_probs(model, meta, test)
        seg = calibrate(model.predict_proba(segment_batch(data, test_rows)), meta["temperature"])
        clip_m = compute_metrics(test.labels, probs, classes, critical)
        plot_confusion(clip_m["confusion_matrix"], classes, REPORTS_DIR / "confusion" / f"{v}_test.png",
                       f"{v}: test clips ({meta['aggregation']} of windows)")
        pred = probs.argmax(1)
        per_source = {}
        for src in sorted(set(test.sources)):
            mask = test.sources == src
            per_source[src] = {"clips": int(mask.sum()), "accuracy": round(float((pred[mask] == test.labels[mask]).mean()), 4)}
        results[v] = {
            "name": meta["name"],
            "test_clip": clip_m,
            "test_segment": compute_metrics(data.y[test_rows], seg, classes, critical),
            "per_source_accuracy": per_source,
            "validation_clip_macro_f1": meta["validation"]["clip"]["macro_f1"],
            "ms_per_window": meta["inference_ms_per_window"],
            "upload_30s_ms": round(upload_latency_ms(model, meta), 1),
            "robustness": {},
        }

    print("hidden-test simulator:")
    for condition in CONDITIONS:
        windows = condition_windows(condition, classes)
        for v, (model, meta) in loaded.items():
            m = compute_metrics(windows.labels, clip_probs(model, meta, windows), classes, critical)
            results[v]["robustness"][condition] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                                                   "critical_recall": m["critical_recall"]}
        print(f"  {condition:<11} " + "  ".join(f"{loaded[v][1]['name']} {results[v]['robustness'][condition]['macro_f1']:.3f}"
                                                for v in versions))
    for v in versions:
        clean = results[v]["test_clip"]["macro_f1"]
        scores = [r["macro_f1"] for r in results[v]["robustness"].values()]
        results[v]["robustness_mean_macro_f1"] = round(float(np.mean(scores)), 4)
        results[v]["robustness_retained"] = round(float(np.mean(scores) / clean), 4) if clean else 0.0

    factory = factory_windows(classes)
    machinery = classes.index(MACHINERY) if MACHINERY in classes else None
    for v, (model, meta) in loaded.items():
        pred = clip_probs(model, meta, factory).argmax(1)
        counts = {classes[i]: int((pred == i).sum()) for i in np.unique(pred)}
        fault_share = float((pred == machinery).mean()) if machinery is not None else 0.0
        results[v]["factory_test"] = {"clips": int(len(pred)), "predicted": counts,
                                      "called_machinery_fault": round(fault_share, 4),
                                      "flagged": fault_share > FACTORY_FLAG}

    chosen, note = select({v: loaded[v][1] for v in versions})
    summary = {"selected": chosen, "selection_note": note, "classes": classes, "critical": critical,
               "models": results, "conditions": DESCRIPTIONS}
    (REPORTS_DIR / "comparison.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (MODELS_DIR / "selected.json").write_text(json.dumps({"python_model": chosen, "rule": __doc__.split("Selection rule")[1].split("Test results")[0].strip()}, indent=1), encoding="utf-8")
    (REPORTS_DIR / "comparison.md").write_text(markdown(summary), encoding="utf-8")
    plot_robustness(summary, REPORTS_DIR / "robustness.png")

    for v in versions:
        print()
        print(format_metrics(results[v]["test_clip"], f"{v}: TEST, per clip"))
    print(f"\nselected: {chosen} ({note})")
    return summary


def markdown(s: dict) -> str:
    models = s["models"]
    names = list(models)
    lines = ["# Model comparison (test set)", "",
             f"Selected: **{s['selected']}** ({s['selection_note']}). Rule: see `src/sonic/models/compare.py`.", "",
             "## Overall", "",
             "| Model | Val macro F1 | Test accuracy | Test macro F1 | Test macro precision | Test macro recall | ECE | Robust macro F1 | Retained | ms / window | 30 s upload (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in names:
        r, t = models[v], models[v]["test_clip"]
        lines.append(f"| {v} | {r['validation_clip_macro_f1']:.3f} | {t['accuracy']:.1%} | {t['macro_f1']:.3f} | "
                     f"{t['macro_precision']:.3f} | {t['macro_recall']:.3f} | {t['ece']:.3f} | "
                     f"{r['robustness_mean_macro_f1']:.3f} | {r['robustness_retained']:.0%} | {r['ms_per_window']:.1f} | {r['upload_30s_ms']:.0f} |")
    lines += ["", "## Critical-event recall (test, target >= 85 %)", "",
              "| Model | " + " | ".join(s["critical"]) + " |", "|---|" + "---|" * len(s["critical"])]
    for v in names:
        cr = models[v]["test_clip"]["critical_recall"]
        lines.append(f"| {v} | " + " | ".join(f"{cr[c]:.0%}{'' if cr[c] >= 0.85 else ' ✗'}" for c in s["critical"]) + " |")
    lines += ["", "## Class-wise F1 (test)", "", "| Class | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for c in s["classes"]:
        lines.append(f"| {c} | " + " | ".join(f"{models[v]['test_clip']['per_class'][c]['f1']:.2f}" for v in names) + " |")
    lines += ["", "## Hidden-test simulator (test macro F1)", "", "| Condition | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    lines.append("| clean | " + " | ".join(f"{models[v]['test_clip']['macro_f1']:.3f}" for v in names) + " |")
    for cond, desc in s["conditions"].items():
        lines.append(f"| {cond}: {desc} | " + " | ".join(f"{models[v]['robustness'][cond]['macro_f1']:.3f}" for v in names) + " |")
    lines += ["", "## Factory-shortcut test", "",
              "60 MIMII normal-operation recordings never used in the dataset, training or augmentation. A model "
              "that learned the fault (not the factory background) should not call them Machinery Fault; the "
              "correct answer is Normal Machinery.", "",
              "| Model | Called Machinery Fault | Predictions |", "|---|---|---|"]
    for v in names:
        f = models[v]["factory_test"]
        flag = " ⚠ flagged" if f["flagged"] else ""
        lines.append(f"| {v} | {f['called_machinery_fault']:.0%}{flag} | "
                     + ", ".join(f"{k}: {n}" for k, n in sorted(f["predicted"].items(), key=lambda kv: -kv[1])) + " |")
    lines += ["", "## Accuracy by source dataset (test)", "", "| Source | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for src in models[names[0]]["per_source_accuracy"]:
        n = models[names[0]]["per_source_accuracy"][src]["clips"]
        lines.append(f"| {src} ({n} clips) | " + " | ".join(f"{models[v]['per_source_accuracy'][src]['accuracy']:.0%}" for v in names) + " |")
    return "\n".join(lines) + "\n"


def plot_robustness(s: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = ["clean", *s["conditions"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for v, r in s["models"].items():
        values = [r["test_clip"]["macro_f1"], *(r["robustness"][c]["macro_f1"] for c in s["conditions"])]
        ax.plot(conditions, values, marker="o", label=v)
    ax.set_ylabel("Test macro F1")
    ax.set_ylim(0, 1)
    ax.set_title("Hidden-test simulator: macro F1 per condition")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
