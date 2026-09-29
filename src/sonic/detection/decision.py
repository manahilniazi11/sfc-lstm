"""The final sound-event decision (SRS Steps 14-19, FR xxxviii-li).

Why the Python model leads: on the untouched test set it reached 86 %
accuracy (GTM: about 46 % on its own held-out samples), so its class is the
candidate. GTM is the independent second opinion: agreement raises the
confidence level, disagreement sends the event to a human.

Every decision records *why*: the alert conditions that passed or failed and
the manual-review reasons, so a result can always be explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compare import ACCEPTABLE, Comparison, Scores, compare
from .rules import BACKGROUND, NON_EVENTS, QUALITY_GRADES, UNKNOWN, RuleSet

# Event statuses (FR lxii). Reviewed and Closed are set later by people.
UPLOADED, CLASSIFIED, UNCERTAIN_STATUS, ALERT_GENERATED, MANUAL_REVIEW, REVIEWED, CLOSED = (
    "Uploaded", "Classified", "Uncertain", "Alert Generated", "Manual Review", "Reviewed", "Closed",
)
STATUSES = [UPLOADED, CLASSIFIED, UNCERTAIN_STATUS, ALERT_GENERATED, MANUAL_REVIEW, REVIEWED, CLOSED]

NO_ALERT, NOT_CONFIRMED, ALERT = "No alert", "Not confirmed", "Alert generated"

# Manual-review reasons, worded as in SRS Step 17.
R_DISAGREE = "Different predictions from the two models"
R_LOW_CONF = "Low confidence"
R_POOR_QUALITY = "Poor audio quality"
R_SIMILAR = "Similar top-class confidence scores"
R_OVERLAP = "Overlapping sounds"
R_UNSUPPORTED = "Unsupported sound pattern"
R_CRITICAL_NO_AGREEMENT = "Critical event without sufficient model agreement"
R_FALSE_ALARM = "Possible false alarm"
R_NO_GTM = "GTM result unavailable"
# Reasons that make the result itself uncertain (FR xxxviii) rather than just worth a second look.
UNCERTAIN_REASONS = {R_DISAGREE, R_LOW_CONF, R_POOR_QUALITY, R_SIMILAR, R_OVERLAP, R_NO_GTM}


@dataclass
class Decision:
    final_class: str
    confidence: float
    confidence_level: str
    severity: str
    alert_status: str
    recommended_action: str
    review_required: bool
    review_reasons: list[str]
    uncertain: bool
    status: str
    comparison: Comparison
    overlapping_classes: list[str] = field(default_factory=list)
    repeated_detections: int = 1
    conditions: dict[str, bool] = field(default_factory=dict)  # alert conditions and whether each passed

    @property
    def alert(self) -> bool:
        return self.alert_status == ALERT


def overlapping_classes(windows: list[Scores], threshold: float) -> list[str]:
    """Event classes that share a window with another event class, each above `threshold` (FR xxxix)."""
    found: set[str] = set()
    for w in windows:
        strong = [c for c, p in w.probs.items() if p >= threshold and c not in NON_EVENTS]
        if len(strong) >= 2:
            found.update(strong)
    return sorted(found)


def longest_run(windows: list[Scores], category: str) -> int:
    """Most consecutive windows whose top class is `category` (repeated detection inside one recording)."""
    best = run = 0
    for w in windows:
        run = run + 1 if w.top == category else 0
        best = max(best, run)
    return best


def decide(
    python: Scores,
    gtm: Scores | None,
    quality_grade: str,
    ruleset: RuleSet,
    windows: list[Scores] | None = None,
    repeated_detections: int | None = None,
    noise_level_dbfs: float | None = None,
) -> Decision:
    """Combine both models, audio quality and the category's rule into the final result.

    `windows` are the Python model's per-window scores for this recording (for
    overlap and repeated detection). Live monitoring passes
    `repeated_detections` instead: how often the class was detected within the
    configured period. `noise_level_dbfs` is the recording's level, used by the
    Background Noise rule.
    """
    t = ruleset.thresholds
    windows = windows or [python]
    comparison = compare(python, gtm, t)
    confidence, margin = python.confidence, python.margin

    unknown = confidence < t.unknown_confidence or quality_grade == "Unusable"
    final = UNKNOWN if unknown else python.top
    rule = ruleset.for_category(final)

    reasons: list[str] = []
    if gtm is None:
        reasons.append(R_NO_GTM)
    elif not comparison.match:
        reasons.append(R_DISAGREE)
    if confidence < t.min_confidence:
        reasons.append(R_LOW_CONF)
    if QUALITY_GRADES.index(quality_grade) >= QUALITY_GRADES.index("Poor"):
        reasons.append(R_POOR_QUALITY)
    if margin < t.top_two_margin:
        reasons.append(R_SIMILAR)
    overlaps = overlapping_classes(windows, t.overlap_threshold)
    if overlaps:
        reasons.append(R_OVERLAP)
    if unknown:
        reasons.append(R_UNSUPPORTED)

    needed = rule.consecutive_detections
    if repeated_detections is None:
        repeated_detections = longest_run(windows, final) if not unknown else 0
        # A 1 s clip has one window: it cannot show the same class twice in a row.
        needed = min(needed, len(windows)) if len(windows) else needed
    # Live monitoring counts repeats across its windows, so a short sound (one
    # model window inside a live window) is never enough on its own.

    severity, raises_alert, action = rule.severity, rule.alert, rule.recommended_action
    if final == BACKGROUND and rule.noise_limit_dbfs is not None and noise_level_dbfs is not None \
            and noise_level_dbfs > rule.noise_limit_dbfs and rule.above_noise_limit:
        severity = rule.above_noise_limit.get("severity", severity)
        raises_alert = rule.above_noise_limit.get("alert", raises_alert)
        action = rule.above_noise_limit.get("recommended_action", action)

    conditions: dict[str, bool] = {}
    alert_status = NO_ALERT
    if raises_alert and not unknown:
        conditions = {
            f"confidence >= {rule.min_confidence:.2f}": confidence >= rule.min_confidence,
            f"top-two margin >= {rule.top_two_margin:.2f}": margin >= rule.top_two_margin,
            f"detected in {needed} consecutive window(s)": repeated_detections >= needed,
            f"audio quality {rule.min_quality} or better": QUALITY_GRADES.index(quality_grade) <= QUALITY_GRADES.index(rule.min_quality),
        }
        if rule.require_model_agreement:
            conditions["both models agree"] = comparison.match
        alert_status = ALERT if all(conditions.values()) else NOT_CONFIRMED
        if alert_status == NOT_CONFIRMED:
            reasons.append(R_FALSE_ALARM)
    if rule.critical and not unknown and not comparison.match:
        reasons.append(R_CRITICAL_NO_AGREEMENT)

    raise_rule = rule.raise_to_critical
    if alert_status == ALERT and raise_rule \
            and confidence >= raise_rule.get("min_confidence", 1.0) \
            and repeated_detections >= raise_rule.get("consecutive_detections", 1):
        severity = "Critical"

    if confidence >= max(0.80, rule.min_confidence) and margin >= t.top_two_margin and comparison.status == ACCEPTABLE:
        level = "High"
    elif confidence >= t.min_confidence:
        level = "Medium"
    else:
        level = "Low"

    reasons = list(dict.fromkeys(reasons))  # keep order, drop repeats
    uncertain = any(r in UNCERTAIN_REASONS for r in reasons)
    if alert_status == ALERT:
        status = ALERT_GENERATED
    elif uncertain:
        status = UNCERTAIN_STATUS
    elif reasons:
        status = MANUAL_REVIEW
    else:
        status = CLASSIFIED

    return Decision(
        final_class=final,
        confidence=round(confidence, 4),
        confidence_level=level,
        severity=severity,
        alert_status=alert_status,
        recommended_action=action,
        review_required=bool(reasons),
        review_reasons=reasons,
        uncertain=uncertain,
        status=status,
        comparison=comparison,
        overlapping_classes=overlaps,
        repeated_detections=repeated_detections,
        conditions=conditions,
    )
