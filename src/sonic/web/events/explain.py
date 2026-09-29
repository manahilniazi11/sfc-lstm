"""Plain-language explanation of a stored event (fixed sentences filled from the stored results; no AI)."""

from __future__ import annotations

from sonic.detection import decision as D

CONSISTENCY_MEANING = {
    "Acceptable Match": "both models chose the same class with similar, sufficient confidence",
    "Weak Match": "both models chose the same class, but at least one was unsure or their confidences differ a lot",
    "Model Disagreement": "the models chose different classes and each was confident",
    "Uncertain Result": "the models chose different classes and at least one of them was unsure (or GTM gave no result)",
}


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def confidence_note(event) -> str:
    """Why the confidence level is High, Medium or Low."""
    if event.confidence_level == "High":
        return "confident, clear winner, and both models agree"
    if event.confidence_level == "Medium":
        if event.consistency != "Acceptable Match":
            return "the Python model is confident, but the two models do not fully agree"
        return "confident, but not above the High level (80%) with a clear margin"
    return "below the minimum confidence: the result may well be wrong"


def summary(event) -> list[str]:
    """A few sentences a non-expert can read: what each model heard, and what the app did about it."""
    windows = event.windows.get("python", [])
    same = sum(1 for w in windows if max(w["scores"], key=w["scores"].get) == event.python_class)
    lines = [
        f"The Python model heard {event.python_class} ({_pct(event.python_confidence)} confidence); "
        f"it was the top class in {same} of {len(windows)} one-second window{'s' if len(windows) != 1 else ''}."
    ]
    if event.gtm_scores:
        verdict = "agrees" if event.class_match else "disagrees"
        lines.append(f"Google Teachable Machine {verdict}: it heard {event.gtm_class} ({_pct(event.gtm_confidence)}). "
                     f"Result: {event.consistency} — {CONSISTENCY_MEANING.get(event.consistency, '')}.")
    else:
        lines.append(f"Google Teachable Machine gave no result ({event.gtm_error or 'unavailable'}), so the result cannot be cross-checked.")

    if event.final_class == "Unknown":
        lines.append("The confidence is too low to name a class, so the sound is reported as Unknown.")
    if event.alert_status == D.ALERT:
        lines.append(f"{event.final_class} has an alert rule and every condition passed, so a {event.severity} alert was raised.")
    elif event.alert_status == D.NOT_CONFIRMED:
        failed = [name for name, ok in event.conditions.items() if not ok]
        lines.append(f"{event.final_class} can raise an alert, but it was not confirmed: {', '.join(failed)} failed.")
    else:
        lines.append(f"{event.final_class} does not raise an alert (severity {event.severity}).")

    if event.review_required:
        lines.append("A reviewer should check it because: " + "; ".join(r.lower() for r in event.review_reasons) + ".")
    else:
        lines.append("No manual review is needed.")
    return lines


def comparison_rows(event) -> list[dict]:
    """Every class with both models' confidence and the gap (SRS Step 11: confidence for all classes)."""
    python = event.python_scores or {}
    gtm = event.gtm_scores or {}
    classes = sorted(set(python) | set(gtm), key=lambda c: max(python.get(c, 0), gtm.get(c, 0)), reverse=True)
    return [{
        "name": c,
        "python": python.get(c),
        "gtm": gtm.get(c),
        "difference": abs(python[c] - gtm[c]) if c in python and c in gtm else None,
        "python_top": c == event.python_class,
        "gtm_top": c == event.gtm_class,
    } for c in classes]
