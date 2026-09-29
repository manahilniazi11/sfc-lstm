"""Metrics (SRS Step 7 selection list), clip-level decisions and confidence calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

EPS = 1e-12
AGGREGATIONS = ("mean", "max", "top3")


# --- calibration --------------------------------------------------------------------

def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Temperature T that makes ``softmax(log p / T)`` best match the true labels (min NLL).

    T > 1 softens over-confident models, T < 1 sharpens under-confident ones.
    The ranking of classes never changes, so accuracy is unaffected.
    """
    logp = np.log(probs + EPS)

    def nll(log_t: float) -> float:
        return -np.mean(np.log(calibrate_logp(logp, np.exp(log_t))[np.arange(len(y)), y] + EPS))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def calibrate(probs: np.ndarray, temperature: float) -> np.ndarray:
    return calibrate_logp(np.log(probs + EPS), temperature)


def calibrate_logp(logp: np.ndarray, temperature: float) -> np.ndarray:
    z = logp / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Average gap between confidence and accuracy: 0.05 means '90 % confident' is right ~85-95 % of the time."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs((pred[in_bin] == y[in_bin]).mean() - conf[in_bin].mean())
    return float(ece)


# --- clip-level decisions ---------------------------------------------------------------

def aggregate(window_probs: np.ndarray, offsets: np.ndarray, method: str) -> np.ndarray:
    """Combine each clip's window probabilities into one distribution per clip.

    mean: average over windows (steady sounds); max: each class's highest
    window (a single short event is not diluted); top3: average of the 3
    windows the model is most sure about. Clips without any non-silent
    window get a uniform distribution.
    """
    n_classes = window_probs.shape[1]
    out = np.full((len(offsets) - 1, n_classes), 1 / n_classes)
    for i in range(len(offsets) - 1):
        p = window_probs[offsets[i] : offsets[i + 1]]
        if len(p) == 0:
            continue
        if method == "mean":
            clip = p.mean(axis=0)
        elif method == "max":
            clip = p.max(axis=0)
        elif method == "top3":
            clip = p[np.argsort(-p.max(axis=1))[:3]].mean(axis=0)
        else:
            raise ValueError(f"unknown aggregation {method}")
        out[i] = clip / clip.sum()
    return out


# --- metrics ---------------------------------------------------------------------------------

def compute_metrics(y: np.ndarray, probs: np.ndarray, classes: list[str], critical: list[str]) -> dict:
    """Everything SRS Step 7 lists, for one set of predictions."""
    pred = probs.argmax(axis=1)
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = {
        c: {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4), "support": int(s)}
        for c, p, r, f, s in zip(classes, precision, recall, f1, support)
    }
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_precision": round(float(precision.mean()), 4),
        "macro_recall": round(float(recall.mean()), 4),
        "macro_f1": round(float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)), 4),
        "per_class": per_class,
        "critical_recall": {c: per_class[c]["recall"] for c in critical if c in per_class},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "ece": round(expected_calibration_error(probs, y), 4),
        "n": int(len(y)),
    }


def plot_confusion(cm: list[list[int]], classes: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if share[i, j] > 0.5 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def format_metrics(m: dict, title: str, target_recall: float = 0.85) -> str:
    lines = [f"{title}: accuracy {m['accuracy']:.1%}, macro F1 {m['macro_f1']:.3f}, ECE {m['ece']:.3f} (n={m['n']})"]
    lines.append(f"  {'class':<24}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>6}")
    for c, v in m["per_class"].items():
        flag = ""
        if c in m["critical_recall"]:
            flag = "  critical OK" if v["recall"] >= target_recall else "  critical BELOW 85%"
        lines.append(f"  {c:<24}{v['precision']:>7.2f}{v['recall']:>8.2f}{v['f1']:>7.2f}{v['support']:>6}{flag}")
    return "\n".join(lines)
