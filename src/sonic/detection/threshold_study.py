"""Choose the decision thresholds from data: `python -m sonic.detection.threshold_study`.

For the selected Python model it measures, on **validation** clips only (the
test split stays untouched), how accurate the model is above and below each
confidence and top-two-margin threshold, and how much of each critical class
an alert threshold would still catch. The values in alert_rules.json were
picked from this report (reports/threshold_study.md).
"""

from __future__ import annotations

import numpy as np

from ..dataset.config import REPO_ROOT
from ..inference.python_model import selected_version
from ..models.compare import clip_probs, load_model
from ..models.windows import load_windows

REPORT = REPO_ROOT / "reports" / "threshold_study.md"
CONFIDENCES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
MARGINS = [0.05, 0.10, 0.15, 0.20, 0.30]
CRITICAL = ["Gunshot", "Glass Breaking", "Panic Scream", "Aggression", "Person Asking for Help"]


def _row(mask: np.ndarray, correct: np.ndarray) -> tuple[float, float, float]:
    rest = correct[~mask]
    return mask.mean(), correct[mask].mean() if mask.any() else float("nan"), rest.mean() if rest.size else float("nan")


def study() -> str:
    version = selected_version()
    model, meta = load_model(version)
    windows = load_windows("validation", meta["classes"])
    probs = clip_probs(model, meta, windows)
    conf, pred = probs.max(1), probs.argmax(1)
    ordered = np.sort(probs, 1)
    margin = ordered[:, -1] - ordered[:, -2]
    correct = pred == windows.labels

    lines = [
        "# Decision-threshold study",
        "",
        f"Model `{version}`, {len(conf)} validation clips (clip decision as in the app). "
        f"Accuracy {correct.mean():.1%}, median top confidence {np.median(conf):.2f}.",
        "",
        "## Top-class confidence",
        "",
        "| Threshold | Clips at or above | Accuracy at or above | Accuracy below |",
        "|---|---|---|---|",
    ]
    for t in CONFIDENCES:
        share, above, below = _row(conf >= t, correct)
        lines.append(f"| {t:.2f} | {share:.0%} | {above:.1%} | {below:.1%} |")
    lines += ["", "## Top-two margin", "", "| Threshold | Clips at or above | Accuracy at or above | Accuracy below |", "|---|---|---|---|"]
    for t in MARGINS:
        share, above, below = _row(margin >= t, correct)
        lines.append(f"| {t:.2f} | {share:.0%} | {above:.1%} | {below:.1%} |")
    lines += ["", "## Critical classes: share of real events that would reach an alert threshold", "",
              "| Class | Recall (any confidence) | at >= 0.45 | at >= 0.50 | at >= 0.60 | at >= 0.70 |", "|---|---|---|---|---|---|"]
    for name in CRITICAL:
        if name not in meta["classes"]:
            lines.append(f"| {name} | not in the model yet | | | | |")
            continue
        ci = meta["classes"].index(name)
        real = windows.labels == ci
        hit = pred[real] == ci
        cells = [f"{np.mean(hit & (conf[real] >= t)):.0%}" for t in (0.45, 0.50, 0.60, 0.70)]
        lines.append(f"| {name} | {hit.mean():.0%} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Chosen values (alert_rules.json)",
        "",
        "- `min_confidence` 0.50: above it clips are clearly more reliable; below it the model is right only about half the time, so those results go to manual review.",
        "- `unknown_confidence` 0.35: below it the model is mostly wrong, so the sound is reported as Unknown.",
        "- `top_two_margin` 0.10: below it accuracy drops sharply (two classes are almost tied).",
        "- Panic Scream and Aggression alert at 0.45: their confidences are lower than the other critical classes, and a missed scream is worse than a reviewed false alarm.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    text = study()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {REPORT}")


if __name__ == "__main__":
    main()
