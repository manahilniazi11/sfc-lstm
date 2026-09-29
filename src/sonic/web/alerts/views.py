from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Case, IntegerField, Value, When
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from sonic.detection.rules import SEVERITIES

from ..accounts.decorators import capability_required
from .context import visible_alerts
from .handling import CLOSED, ActionRefused, act, available_actions, escalate_overdue
from .models import Alert, AlertRule

TABS = {
    "attention": ("Needs attention", Alert.OPEN),
    "handling": ("Being handled", [Alert.Status.ACKNOWLEDGED]),
    "closed": ("Dismissed or resolved", list(CLOSED)),
    "all": ("All alerts", None),
}
# Critical first, then High, ... (the order people should work through them).
SEVERITY_RANK = Case(*[When(severity=s, then=Value(i)) for i, s in enumerate(reversed(SEVERITIES))],
                     default=Value(len(SEVERITIES)), output_field=IntegerField())


@login_required
def alert_list(request):
    escalate_overdue()
    tab = request.GET.get("tab") if request.GET.get("tab") in TABS else "attention"
    alerts = visible_alerts(request.user)
    counts = {key: (alerts.filter(status__in=statuses) if statuses else alerts).count() for key, (_, statuses) in TABS.items()}
    statuses = TABS[tab][1]
    if statuses:
        alerts = alerts.filter(status__in=statuses)
    filters = {"severity": request.GET.get("severity", ""), "category": request.GET.get("category", ""),
               "alert_type": request.GET.get("alert_type", "")}
    alerts = alerts.filter(**{k: v for k, v in filters.items() if v})
    if tab == "attention":
        alerts = alerts.annotate(rank=SEVERITY_RANK).order_by("rank", "-created_at")
    page = Paginator(alerts, 25).get_page(request.GET.get("page"))
    return render(request, "alerts/list.html", {
        "page": page, "tab": tab, "tabs": [(k, label, counts[k]) for k, (label, _) in TABS.items()],
        "filters": filters, "severities": list(reversed(SEVERITIES)),
        "categories": AlertRule.objects.filter(alert=True).values_list("category", flat=True).order_by("category"),
        "alert_types": sorted(set(AlertRule.objects.values_list("alert_type", flat=True))),
    })


def _get_alert(user, code: str) -> Alert:
    return get_object_or_404(visible_alerts(user), code=code)


@login_required
def alert_detail(request, code: str):
    alert = _get_alert(request.user, code)
    event = alert.event
    return render(request, "alerts/detail.html", {
        "alert": alert, "event": event, "record": event.record,
        "actions": alert.actions.select_related("user"),
        "available": available_actions(alert) if request.user.can("handle_alerts") else [],
        "detections": alert.actions.filter(kind="note", user__isnull=True).count() + 1,
    })


@require_POST
@capability_required("handle_alerts")
def alert_action(request, code: str, action: str):
    alert = _get_alert(request.user, code)
    try:
        act(alert, action, request.user, request.POST.get("note", ""), request=request)
    except ActionRefused as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, f"{alert.code}: {'note added' if action == 'note' else alert.get_status_display().lower()}.")
    return redirect("alerts:detail", code=alert.code)
