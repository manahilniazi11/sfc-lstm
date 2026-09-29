"""Numbers for the dashboards and the analytics page (FR lxiii, lxv, lxviii; SRS Step 18).

Every function takes a SoundEvent queryset that is already limited to what
the viewer may see (and to any filters), so the same code serves a normal
user's own events and an administrator's whole system.

"Effective" class and severity mean the reviewer's decision when there is
one, otherwise the automatic result.
"""

from __future__ import annotations

from datetime import timedelta
from statistics import median

from django.db.models import Avg, Count, Min, Q, Value
from django.db.models.functions import Coalesce, NullIf, TruncDate
from django.utils import timezone

from sonic.detection import decision as D
from sonic.detection.rules import QUALITY_GRADES, SEVERITIES

from ..alerts.models import Alert, AlertAction, current_ruleset

EFFECTIVE_CLASS = Coalesce(NullIf("reviewed_class", Value("")), "final_class")
EFFECTIVE_SEVERITY = Coalesce(NullIf("reviewed_severity", Value("")), "severity")
POOR = ["Poor", "Unusable"]


def with_effective(events):
    return events.annotate(eff_class=EFFECTIVE_CLASS, eff_severity=EFFECTIVE_SEVERITY)


def critical_classes() -> list[str]:
    return sorted(c for c, rule in current_ruleset().rules.items() if rule.critical)


def overview(events, alerts) -> dict:
    """The headline figures (administrator dashboard, FR lxv)."""
    events = with_effective(events)
    totals = events.aggregate(
        total=Count("id"),
        avg_confidence=Avg("python_confidence"),
        compared=Count("id", filter=~Q(gtm_class="")),
        disagreements=Count("id", filter=Q(class_match=False) & ~Q(gtm_class="")),
        poor_quality=Count("id", filter=Q(record__quality_grade__in=POOR)),
        critical=Count("id", filter=Q(eff_severity="Critical")),
        waiting=Count("id", filter=Q(review_required=True) & ~Q(status__in=[D.REVIEWED, D.CLOSED])),
    )
    totals["disagreement_rate"] = totals["disagreements"] / totals["compared"] if totals["compared"] else None
    totals["critical_alerts"] = alerts.filter(critical=True).count()
    totals["open_alerts"] = alerts.filter(status__in=Alert.OPEN).count()
    return totals


def by_category(events) -> list[tuple[str, int]]:
    rows = with_effective(events).values("eff_class").annotate(n=Count("id")).order_by("-n", "eff_class")
    return [(r["eff_class"], r["n"]) for r in rows]


def daily(events, days: int = 14) -> dict:
    """Events and critical events per day for the last `days` days, zero-filled (detection trends)."""
    today = timezone.localdate()
    start = today - timedelta(days=days - 1)
    rows = (with_effective(events).filter(created_at__date__gte=start)
            .annotate(day=TruncDate("created_at", tzinfo=timezone.get_current_timezone()))
            .values("day").annotate(n=Count("id"), critical=Count("id", filter=Q(eff_severity="Critical"))))
    found = {r["day"]: r for r in rows}
    dates = [start + timedelta(days=i) for i in range(days)]
    return {
        "labels": [d.strftime("%d %b") for d in dates],
        "events": [found.get(d, {}).get("n", 0) for d in dates],
        "critical": [found.get(d, {}).get("critical", 0) for d in dates],
    }


def critical_frequency(events, alerts) -> list[dict]:
    """For each critical class: detections and the alerts they raised (FR lxviii)."""
    counts = dict(by_category(events))
    raised = dict(alerts.values_list("category").annotate(n=Count("id")).values_list("category", "n"))
    return [{"category": c, "events": counts.get(c, 0), "alerts": raised.get(c, 0)} for c in critical_classes()]


def confidence_histogram(events, bins: int = 10) -> dict:
    """How many results fall in each confidence band, for both models (confidence distribution)."""
    python, gtm = [0] * bins, [0] * bins
    for p, g in events.values_list("python_confidence", "gtm_confidence"):
        python[min(int(p * bins), bins - 1)] += 1
        if g is not None:
            gtm[min(int(g * bins), bins - 1)] += 1
    step = 100 // bins
    return {"labels": [f"{i * step}–{(i + 1) * step}%" for i in range(bins)], "python": python, "gtm": gtm}


def quality_distribution(events) -> list[tuple[str, int]]:
    counts = dict(events.values_list("record__quality_grade").annotate(n=Count("id")).values_list("record__quality_grade", "n"))
    return [(g, counts.get(g, 0)) for g in QUALITY_GRADES]


def disagreement_by_class(events) -> list[dict]:
    """Share of results per (automatic) class where GTM predicted a different class."""
    rows = (events.exclude(gtm_class="").values("final_class")
            .annotate(n=Count("id"), differ=Count("id", filter=Q(class_match=False))).order_by("final_class"))
    return [{"category": r["final_class"], "n": r["n"], "differ": r["differ"], "rate": r["differ"] / r["n"]}
            for r in rows]


def errors_by_class(events, alerts) -> dict:
    """False positives and false negatives per class, from the three sources the app has.

    - Reviewer decisions: a correction away from class X is a false positive
      for X; a correction *to* X is a false negative for X.
    - Dismissed alerts: an alert for X a person dismissed as a false alarm.
    - Dataset labels: uploads that are byte-identical to a labelled dataset
      clip, compared with each model's own prediction.
    """
    classes = sorted(current_ruleset().rules)
    table = {c: {"category": c, "confirmed": 0, "fp": 0, "fn": 0, "dismissed": 0,
                 "py_fp": 0, "py_fn": 0, "gtm_fp": 0, "gtm_fn": 0} for c in classes}

    def row(c):
        return table.setdefault(c, {"category": c, "confirmed": 0, "fp": 0, "fn": 0, "dismissed": 0,
                                    "py_fp": 0, "py_fn": 0, "gtm_fp": 0, "gtm_fn": 0})

    for automatic, reviewed in events.exclude(reviewed_class="").values_list("final_class", "reviewed_class"):
        if automatic == reviewed:
            row(automatic)["confirmed"] += 1
        else:
            row(automatic)["fp"] += 1
            row(reviewed)["fn"] += 1
    for category, n in (alerts.filter(status=Alert.Status.DISMISSED).values_list("category")
                        .annotate(n=Count("id")).values_list("category", "n")):
        row(category)["dismissed"] = n

    labelled = events.exclude(record__reference_class="").values_list("record__reference_class", "python_class", "gtm_class")
    evaluation = {"n": 0, "python_correct": 0, "gtm_n": 0, "gtm_correct": 0}
    for truth, python, gtm in labelled:
        evaluation["n"] += 1
        if python == truth:
            evaluation["python_correct"] += 1
        else:
            row(python)["py_fp"] += 1
            row(truth)["py_fn"] += 1
        if gtm:
            evaluation["gtm_n"] += 1
            if gtm == truth:
                evaluation["gtm_correct"] += 1
            else:
                row(gtm)["gtm_fp"] += 1
                row(truth)["gtm_fn"] += 1
    evaluation["python_accuracy"] = evaluation["python_correct"] / evaluation["n"] if evaluation["n"] else None
    evaluation["gtm_accuracy"] = evaluation["gtm_correct"] / evaluation["gtm_n"] if evaluation["gtm_n"] else None
    return {"rows": list(table.values()), "evaluation": evaluation}


HUMAN_ACTIONS = [AlertAction.Kind.ACKNOWLEDGED, AlertAction.Kind.DISMISSED, AlertAction.Kind.ESCALATED,
                 AlertAction.Kind.RESOLVED]


def alert_response(alerts) -> list[dict]:
    """Per severity: how many alerts, how fast a person first acted, how many escalated or were dismissed."""
    rows = alerts.annotate(first_action=Min("actions__created_at", filter=Q(actions__kind__in=HUMAN_ACTIONS,
                                                                                 actions__user__isnull=False)),
                           auto=Count("actions", filter=Q(actions__kind=AlertAction.Kind.AUTO_ESCALATED)))
    result = []
    for severity in reversed(SEVERITIES):
        subset = [a for a in rows if a.severity == severity]
        if not subset:
            continue
        minutes = [(a.first_action - a.created_at).total_seconds() / 60 for a in subset if a.first_action]
        result.append({
            "severity": severity,
            "alerts": len(subset),
            "handled": len(minutes),
            "median_minutes": median(minutes) if minutes else None,
            "auto_escalated": sum(1 for a in subset if a.auto),
            "dismissed": sum(1 for a in subset if a.status == Alert.Status.DISMISSED),
            "open": sum(1 for a in subset if a.status in Alert.OPEN),
        })
    return result
