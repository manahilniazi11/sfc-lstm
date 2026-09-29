"""Step 3: run the Python model on the same clips, apply the app's decision and write the report.

For every recording the report lists what SRS deliverable 6 asks for: Audio
ID, file name, actual class, both models' prediction and confidence for
every class, class-match status, top-class confidence difference, top-two
margins, audio quality, severity, alert status, manual-review status, the
final decision and whether it was correct, and an explanation for each
major disagreement. The summary compares the two models over all of them.

The Python side is the web app's upload path without the database: the
same decoding, quality check, classifier and `sonic.detection.decide`, with
the rules and thresholds in alert_rules/alert_rules.json.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ..dataset.config import DEFAULT_OUT_DIR, REPO_ROOT
from ..detection import decision as D
from ..detection.compare import Scores
from ..detection.rules import load_rules
from ..inference import audio_io
from ..inference.python_model import get_classifier
from ..quality.analyze import assess
from .runner import RESULTS, RUN_DIR, test_clips

REPORT_MD = REPO_ROOT / "reports" / "model_comparison.md"
REPORT_CSV = REPO_ROOT / "reports" / "model_comparison.csv"
CONFIDENT = 0.60  # a disagreement is "major" when a model was at least this sure, or a critical class is involved

# Why two classes get confused, written from how the sounds are made (used in the explanations).
CONFUSABLE = {
    frozenset({"Gunshot", "Glass Breaking"}): "both are short, broadband impulses with a sharp attack",
    frozenset({"Machinery Fault", "Normal Machinery"}): "they come from the same kinds of machines; a fault "
                                                        "differs only by subtle rhythmic or tonal changes",
    frozenset({"Panic Scream", "Aggression"}): "both are loud, strained human voices; angry shouting and "
                                               "fearful screaming overlap in pitch and loudness",
    frozenset({"Alarm or Siren", "Vehicle Horn"}): "both are loud tonal signals with strong harmonics",
    frozenset({"Alarm or Siren", "Panic Scream"}): "both are sustained, high-pitched sounds",
    frozenset({"Animal Sound", "Panic Scream"}): "animal calls such as screeches share pitch and loudness with screams",
    frozenset({"Animal Sound", "Aggression"}): "barking and growling resemble shouting",
    frozenset({"Gunshot", "Machinery Fault"}): "impacts and knocks from machines are impulsive like shots",
}
BACKGROUND_NOTE = "a quiet or distant event sinks into the background, and busy background noise can resemble an event"
CHANNEL_NOTE = ("GTM learned from the training audio played through the speaker and re-recorded (Stereo Mix), "
                "the Python model from the original files, so each relies on different cues")


@dataclass
class Row:
    audio_id: str
    filename: str
    actual: str
    duration_s: float
    quality: str = ""
    quality_issues: list[str] = field(default_factory=list)
    python: dict[str, float] | None = None
    gtm: dict[str, float] | None = None
    gtm_error: str = ""
    python_windows: list[str] = field(default_factory=list)  # top class of each Python window
    decision: D.Decision | None = None
    rejected: str = ""  # why the app would refuse the file (then there is no decision)
    explanation: str = ""

    @property
    def python_class(self):
        return max(self.python, key=self.python.get) if self.python else ""

    @property
    def gtm_class(self):
        return max(self.gtm, key=self.gtm.get) if self.gtm else ""

    @property
    def final(self):
        return self.decision.final_class if self.decision else "Rejected"


def _gtm_scores(result: dict) -> tuple[dict[str, float] | None, str]:
    if not result or result.get("error"):
        return None, (result or {}).get("error", "no GTM result")
    labels, clip = result.get("labels"), result.get("clip")
    if not labels or not clip or len(labels) != len(clip):
        return None, "invalid GTM result"
    return {label: float(p) for label, p in zip(labels, clip)}, ""


def analyse(gtm_results: dict, per_class: int | None = None, progress=print) -> list[Row]:
    by_id = {r["audio_id"]: r["gtm"] for r in gtm_results["results"]}
    rules, classifier = load_rules(), get_classifier()
    clips = [c for c in test_clips(per_class) if c["audio_id"] in by_id]
    rows = []
    for i, clip in enumerate(clips, 1):
        row = Row(clip["audio_id"], Path(clip["filename"]).name, clip["class_name"], float(clip["duration_s"]))
        row.gtm, row.gtm_error = _gtm_scores(by_id[clip["audio_id"]])
        audio, sr, _ = audio_io.decode(DEFAULT_OUT_DIR / clip["filename"])
        quality = assess(audio, sr)
        row.quality, row.quality_issues = quality.grade, quality.issues
        if not quality.usable:
            row.rejected = "unusable audio: " + "; ".join(quality.issues)
        else:
            result = classifier.analyse(audio, sr)
            if result.clip_probs is None:
                row.rejected = "every window is silent"
            else:
                row.python = result.clip_scores()
                windows = result.window_scores()
                row.python_windows = [max(w, key=w.get) for w in windows]
                row.decision = D.decide(Scores(row.python), Scores(row.gtm) if row.gtm else None, quality.grade, rules,
                                        windows=[Scores(w) for w in windows],
                                        noise_level_dbfs=quality.metrics.get("active_level_dbfs"))
        row.explanation = explain(row, rules) if is_major(row, rules) else ""
        rows.append(row)
        if i % 50 == 0 or i == len(clips):
            progress(f"  {i} / {len(clips)} clips")
    return rows


def is_major(row: Row, rules) -> bool:
    """Different classes, and a model was confident or a critical class is involved."""
    if not row.python or not row.gtm or row.python_class == row.gtm_class:
        return False
    critical = any(rules.for_category(c).critical for c in (row.python_class, row.gtm_class, row.actual))
    return critical or row.python[row.python_class] >= CONFIDENT or row.gtm[row.gtm_class] >= CONFIDENT


def explain(row: Row, rules) -> str:
    """A plain-language explanation built only from this recording's facts and the class pair."""
    p, g = row.python_class, row.gtm_class
    parts = [f"Python heard {p} ({row.python[p]:.0%}), GTM heard {g} ({row.gtm[g]:.0%}); it was {row.actual}"]
    right = [name for name, cls in (("Python", p), ("GTM", g)) if cls == row.actual]
    parts[0] += f", so {right[0]} was right." if right else ", so neither model was right."
    pair = frozenset({p, g})
    if pair in CONFUSABLE:
        parts.append(f"{p} and {g} are easy to confuse: {CONFUSABLE[pair]}.")
    elif "Background Noise" in pair:
        parts.append(f"Background Noise is involved: {BACKGROUND_NOTE}.")
    else:
        parts.append(f"{CHANNEL_NOTE[0].upper()}{CHANNEL_NOTE[1:]}.")
    mixed = Counter(row.python_windows)
    if len(mixed) > 1:
        parts.append("The recording is not uniform: Python's one-second windows heard "
                     + ", ".join(f"{c} in {n}" for c, n in mixed.most_common(3)) + ".")
    if row.quality != "Good" and row.quality_issues:
        parts.append(f"Audio quality was {row.quality} ({'; '.join(row.quality_issues)}).")
    if row.gtm[g] < 0.5:
        parts.append(f"GTM was unsure (top class only {row.gtm[g]:.0%}).")
    if row.decision:
        if row.decision.alert:
            parts.append(f"The app raised a {row.decision.severity} alert for {row.decision.final_class}.")
        if row.decision.review_required:
            parts.append("The app sent it to manual review (" + "; ".join(row.decision.review_reasons).lower() + ").")
    return " ".join(parts)


# --- summary ---

def _metrics(actual, predicted, classes):
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=classes, zero_division=0)
    accuracy = float(np.mean([a == p for a, p in zip(actual, predicted)]))
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1)), "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "per_class": {c: (precision[i], recall[i], f1[i], int(support[i])) for i, c in enumerate(classes)}}


def summarise(rows: list[Row], rules) -> dict:
    scored = [r for r in rows if r.python]
    classes = sorted({r.actual for r in rows})
    actual = [r.actual for r in scored]
    python = [r.python_class for r in scored]
    gtm = [r.gtm_class or "(no result)" for r in scored]
    final = [r.final for r in scored]
    both = [r for r in scored if r.gtm]
    agree = [r for r in both if r.python_class == r.gtm_class]
    disagree = [r for r in both if r.python_class != r.gtm_class]
    critical = [c for c in classes if rules.for_category(c).critical]
    alerting = {c for c in classes if rules.for_category(c).alert}
    alerts = [r for r in scored if r.decision.alert]
    missed = [r for r in scored if r.actual in critical and not r.decision.alert and not r.decision.review_required]
    return {
        "n": len(rows), "scored": len(scored), "rejected": [r for r in rows if r.rejected],
        "gtm_errors": [r for r in rows if r.gtm_error], "classes": classes, "critical": critical,
        "python": _metrics(actual, python, classes), "gtm": _metrics(actual, gtm, classes),
        "final": _metrics(actual, final, classes),
        "confusion_python": confusion_matrix(actual, python, labels=classes),
        "confusion_gtm": confusion_matrix(actual, gtm, labels=classes),
        "agree": len(agree), "compared": len(both),
        "agree_correct": sum(r.python_class == r.actual for r in agree),
        "disagree": len(disagree),
        "disagree_python_right": sum(r.python_class == r.actual for r in disagree),
        "disagree_gtm_right": sum(r.gtm_class == r.actual for r in disagree),
        "major": [r for r in rows if r.explanation],
        "consistency": Counter(r.decision.comparison.status for r in scored),
        "consistency_correct": Counter(r.decision.comparison.status for r in scored if r.final == r.actual),
        "confidence_difference": float(np.mean([r.decision.comparison.confidence_difference for r in both])) if both else None,
        "python_conf_right": _mean(r.python[r.python_class] for r in scored if r.python_class == r.actual),
        "python_conf_wrong": _mean(r.python[r.python_class] for r in scored if r.python_class != r.actual),
        "gtm_conf_right": _mean(r.gtm[r.gtm_class] for r in both if r.gtm_class == r.actual),
        "gtm_conf_wrong": _mean(r.gtm[r.gtm_class] for r in both if r.gtm_class != r.actual),
        "review": sum(r.decision.review_required for r in scored),
        "review_wrong": sum(r.decision.review_required for r in scored if r.final != r.actual),
        "wrong": sum(r.final != r.actual for r in scored),
        "alerts": len(alerts), "alerts_true": sum(r.actual == r.decision.final_class for r in alerts),
        "alerting_actual": sum(r.actual in alerting for r in scored),
        "missed_critical": missed,
        "unknown": sum(r.final == D.UNKNOWN for r in scored),
    }


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


# --- writing ---

def pct(x, digits=1):
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def write_csv(rows: list[Row], path: Path = REPORT_CSV) -> None:
    classes = sorted({r.actual for r in rows} | {c for r in rows for c in (r.python or {})})
    header = ["Audio ID", "Filename", "Actual class", "Duration (s)", "Python predicted class", "Python confidence",
              "Python top-two margin", "GTM predicted class", "GTM confidence", "GTM top-two margin", "GTM error",
              "Class match", "Top-class confidence difference", "Consistency status", "Audio quality", "Quality issues",
              "Final decision", "Correct", "Python correct", "GTM correct", "Confidence level", "Severity",
              "Alert status", "Manual review", "Review reasons", "Explanation of major disagreement"]
    header += [f"Python: {c}" for c in classes] + [f"GTM: {c}" for c in classes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        out = csv.writer(f)
        out.writerow(header)
        for r in rows:
            d = r.decision
            c = d.comparison if d else None
            out.writerow([
                r.audio_id, r.filename, r.actual, f"{r.duration_s:.2f}", r.python_class,
                _num(r.python and r.python[r.python_class]), _num(c and c.python_margin), r.gtm_class,
                _num(r.gtm and r.gtm[r.gtm_class]), _num(c and c.gtm_margin), r.gtm_error,
                "" if not (r.python and r.gtm) else ("Yes" if r.python_class == r.gtm_class else "No"),
                _num(c and c.confidence_difference), c.status if c else "", r.quality, "; ".join(r.quality_issues),
                r.final, _yes(r.final == r.actual), _yes(r.python_class == r.actual), _yes(r.gtm_class == r.actual),
                d.confidence_level if d else "", d.severity if d else "", d.alert_status if d else r.rejected,
                ("Required" if d.review_required else "Not required") if d else "", "; ".join(d.review_reasons) if d else "",
                r.explanation,
            ] + [_num((r.python or {}).get(k)) for k in classes] + [_num((r.gtm or {}).get(k)) for k in classes])


def _num(x):
    return "" if x is None or x == "" else f"{float(x):.4f}"


def _yes(flag):
    return "Yes" if flag else "No"


def write_markdown(rows: list[Row], s: dict, meta: dict, path: Path = REPORT_MD) -> None:
    py, gt, fi = s["python"], s["gtm"], s["final"]
    lines = [
        "# Model prediction and confidence comparison",
        "",
        f"The Python model (**{meta['python_version']}**) and Google Teachable Machine (**{meta['gtm_version']}**) "
        f"classified the same **{s['n']} unseen test recordings** ({len(s['classes'])} classes, "
        f"{min(Counter(r.actual for r in rows).values())} or more per class) independently. Each recording then went "
        "through the app's decision (audio quality, alert rules, severity, manual review) exactly as an upload does.",
        "",
        f"- Test split only: none of these recordings, nor any segment of them, was used to train or tune either model.",
        f"- GTM ran in {meta['browser']} through the app's own `static/js/gtm.js` ({meta['gtm_seconds']:.0f} s for all clips); "
        f"the Python model on the server ({meta['python_seconds']:.0f} s). Neither saw the other's result.",
        f"- Rules and thresholds: `alert_rules/alert_rules.json`. Generated {meta['generated']} by "
        "`python -m sonic.comparison report`; every value per recording, including the confidence for every class, "
        "is in [`model_comparison.csv`](model_comparison.csv).",
        "- Person Asking for Help is not in this report: neither model has been trained on it yet (its recordings "
        "are still being made), so it has no test recordings either.",
        "",
        "## Overall comparison summary",
        "",
        "| | Python model | GTM | Final decision (app) |",
        "|---|---|---|---|",
        f"| Accuracy | {pct(py['accuracy'])} | {pct(gt['accuracy'])} | {pct(fi['accuracy'])} |",
        f"| Macro F1 | {py['macro_f1']:.3f} | {gt['macro_f1']:.3f} | {fi['macro_f1']:.3f} |",
        f"| Macro precision | {py['macro_precision']:.3f} | {gt['macro_precision']:.3f} | {fi['macro_precision']:.3f} |",
        f"| Macro recall | {py['macro_recall']:.3f} | {gt['macro_recall']:.3f} | {fi['macro_recall']:.3f} |",
        f"| Mean top-class confidence when right | {pct(s['python_conf_right'])} | {pct(s['gtm_conf_right'])} | |",
        f"| Mean top-class confidence when wrong | {pct(s['python_conf_wrong'])} | {pct(s['gtm_conf_wrong'])} | |",
        "",
    ]
    lines += summary_sentences(s)
    lines += ["", "## Agreement between the models", "",
              f"| Consistency status | Recordings | Final decision correct |", "|---|---|---|"]
    for status in ["Acceptable Match", "Weak Match", "Model Disagreement", "Uncertain Result"]:
        n = s["consistency"].get(status, 0)
        lines.append(f"| {status} | {n} | {pct(s['consistency_correct'].get(status, 0) / n) if n else '-'} |")
    lines += ["",
              f"Mean top-class confidence difference |Python - GTM|: **{pct(s['confidence_difference'])}**.", "",
              "## Per-class results (test recordings)", "",
              "| Class | n | Python precision | Python recall | Python F1 | GTM precision | GTM recall | GTM F1 | Final F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in s["classes"]:
        p, g, f = py["per_class"][c], gt["per_class"][c], fi["per_class"][c]
        mark = " (critical)" if c in s["critical"] else ""
        lines.append(f"| {c}{mark} | {p[3]} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} | {g[0]:.2f} | {g[1]:.2f} | {g[2]:.2f} | {f[2]:.2f} |")
    for title, matrix in [("Python model", s["confusion_python"]), ("GTM", s["confusion_gtm"])]:
        short = [c.split()[0][:7] for c in s["classes"]]
        lines += ["", f"### Confusion matrix: {title}", "", "Rows: actual class. Columns: predicted class.", "",
                  "| Actual \\ predicted | " + " | ".join(short) + " |", "|---" * (len(short) + 1) + "|"]
        for c, counts in zip(s["classes"], matrix):
            lines.append(f"| {c} | " + " | ".join(str(v) if v else "·" for v in counts) + " |")
    lines += ["", "## Decision outcomes", "",
              f"- Alerts raised: **{s['alerts']}**, of which {s['alerts_true']} for the recording's actual class "
              f"({pct(s['alerts_true'] / s['alerts']) if s['alerts'] else '-'} alert precision).",
              f"- Sent to manual review: **{s['review']}** ({pct(s['review'] / s['scored'])}); these include "
              f"{s['review_wrong']} of the {s['wrong']} recordings whose final decision was wrong, so a reviewer would see "
              f"{pct(s['review_wrong'] / s['wrong']) if s['wrong'] else '-'} of the mistakes.",
              f"- Reported as Unknown: {s['unknown']}.",
              f"- Critical-class recordings with neither an alert nor a review: **{len(s['missed_critical'])}**"
              + (" (" + ", ".join(f"{r.audio_id} heard as {r.final}" for r in s["missed_critical"][:10])
                 + (", ..." if len(s["missed_critical"]) > 10 else "") + ")." if s["missed_critical"] else ".")]
    if s["rejected"]:
        lines.append(f"- Refused by the app as unusable: {', '.join(r.audio_id + ' (' + r.rejected + ')' for r in s['rejected'])}.")
    if s["gtm_errors"]:
        lines.append(f"- GTM gave no result for: {', '.join(r.audio_id + ' (' + r.gtm_error + ')' for r in s['gtm_errors'])}.")
    lines += ["", f"## Explanations of major disagreements ({len(s['major'])})", "",
              f"A disagreement is major when the models chose different classes and either model was at least "
              f"{CONFIDENT:.0%} sure, or a critical class (predicted or actual) is involved.", ""]
    for r in s["major"]:
        lines.append(f"- **{r.audio_id}** ({r.filename}): {r.explanation}")
    lines += ["", "## Every recording", "",
              "Confidence for every class of both models is in the CSV. Correct = the final decision equals the actual class.", "",
              "| Audio ID | Actual | Python | GTM | Match | Conf. diff | Margins (Py / GTM) | Quality | Severity | Alert | Review | Final | Correct |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r.decision
        if not d:
            lines.append(f"| {r.audio_id} | {r.actual} | - | - | - | - | - | {r.quality} | - | {r.rejected} | - | Rejected | No |")
            continue
        c = d.comparison
        gtm_cell = f"{r.gtm_class} {pct(r.gtm[r.gtm_class], 0)}" if r.gtm else "no result"
        lines.append(
            f"| {r.audio_id} | {r.actual} | {r.python_class} {pct(r.python[r.python_class], 0)} | {gtm_cell} | "
            f"{'Yes' if c.match else 'No'} | {pct(c.confidence_difference, 0)} | {pct(c.python_margin, 0)} / {pct(c.gtm_margin, 0)} | "
            f"{r.quality} | {d.severity} | {d.alert_status} | {'Yes' if d.review_required else 'No'} | {d.final_class} | "
            f"{'Yes' if d.final_class == r.actual else '**No**'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_sentences(s: dict) -> list[str]:
    py, gt = s["python"], s["gtm"]
    better = "Python model" if py["macro_f1"] >= gt["macro_f1"] else "GTM"
    gap = abs(py["accuracy"] - gt["accuracy"])
    agree_rate = s["agree"] / s["compared"] if s["compared"] else 0
    lines = [
        f"- The **{better}** is clearly the stronger classifier: {pct(py['accuracy'])} against {pct(gt['accuracy'])} "
        f"accuracy ({pct(gap)} apart), macro F1 {py['macro_f1']:.3f} against {gt['macro_f1']:.3f}. This is why the app "
        "takes its class from the Python model and uses GTM as an independent second opinion.",
        f"- The models chose the same class for {s['agree']} of {s['compared']} recordings ({pct(agree_rate)}). When they "
        f"agreed, that class was right {pct(s['agree_correct'] / s['agree']) if s['agree'] else '-'} of the time.",
        f"- They disagreed on {s['disagree']}: Python was right in {s['disagree_python_right']}, GTM in "
        f"{s['disagree_gtm_right']}, neither in {s['disagree'] - s['disagree_python_right'] - s['disagree_gtm_right']}.",
    ]
    if s["python_conf_right"] and s["python_conf_wrong"]:
        lines.append(f"- Confidence is informative for Python (mean {pct(s['python_conf_right'])} when right, "
                     f"{pct(s['python_conf_wrong'])} when wrong), which is what makes the low-confidence review rule work.")
    return lines


def run(per_class: int | None = None, run_dir: Path = RUN_DIR) -> dict:
    gtm_results = json.loads((run_dir / RESULTS).read_text(encoding="utf-8"))
    from ..gtm.model_info import model_info
    from ..inference.python_model import selected_version

    if gtm_results.get("version") != model_info()["version"]:
        raise SystemExit(f"GTM results are from {gtm_results.get('version')}, but gtm_model/ is "
                         f"{model_info()['version']}: run the GTM page again.")
    started = time.perf_counter()
    rows = analyse(gtm_results, per_class)
    python_seconds = time.perf_counter() - started
    rules = load_rules()
    s = summarise(rows, rules)
    agent = gtm_results.get("user_agent", "")
    browser = "Chromium " + agent.split("Chrome/")[1].split(".")[0] if "Chrome/" in agent else "a browser"
    meta = {"python_version": selected_version(), "gtm_version": gtm_results["version"], "browser": browser,
            "gtm_seconds": gtm_results.get("seconds", 0), "python_seconds": python_seconds,
            "generated": time.strftime("%Y-%m-%d")}
    write_csv(rows)
    write_markdown(rows, s, meta)
    return s
