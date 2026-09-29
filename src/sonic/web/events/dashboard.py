"""The dashboard (home page) and the analytics page (FR lxiii, lxv, lxviii; SRS Step 18).

Everyone gets the dashboard of what they may see: a normal user their own
recordings, operators the whole system. Users who may see analytics
(reviewers, administrators) also get the system overview charts.
"""

from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from sonic.detection import decision as D

from ..accounts.decorators import capability_required
from ..alerts.context import visible_alerts
from ..alerts.models import AdminNotice, Alert
from . import search, stats
from .models import AudioRecord, MonitoringSession
from .views import visible_events


def columns(pairs) -> dict:
    """[(label, value), ...] as {"labels": [...], "values": [...]} for a chart."""
    return {"labels": [label for label, _ in pairs], "values": [value for _, value in pairs]}


@login_required
def home(request):
    user = request.user
    events = visible_events(user).select_related("record__session")
    alerts = visible_alerts(user)
    records = AudioRecord.objects.select_related("owner", "event").filter(event__isnull=False)
    if not user.can("view_all_events"):
        records = records.filter(owner=user)
    context = {
        "totals": stats.overview(events, alerts),
        "recent_uploads": records.exclude(source=AudioRecord.Source.LIVE)[:6],
        "latest": events[:6],
        "critical": stats.with_effective(events).filter(eff_severity__in=["Critical", "High"])[:6],
        "open_alerts_list": alerts.filter(status__in=Alert.UNRESOLVED)[:6],
        "quality_warnings": records.filter(quality_grade__in=stats.POOR)[:6],
        "waiting": events.filter(review_required=True).exclude(status__in=[D.REVIEWED, D.CLOSED])[:6],
        "live_session": MonitoringSession.objects.filter(user=user, ended_at__isnull=True).first(),
    }
    if user.can("administer"):
        context["notices"] = AdminNotice.objects.filter(acknowledged_at__isnull=True)[:5]
    if user.can("analytics"):
        context["trend"] = stats.daily(events)
        context["categories"] = stats.by_category(events)
        context["category_chart"] = columns(context["categories"])
    return render(request, "dashboard/home.html", context)


ANALYTICS_FILTERS = ["date_from", "date_to", "source", "category", "user"]


@capability_required("analytics")
def analytics(request):
    form = search.EventFilterForm(request.GET or None, user=request.user)
    events = search.apply(visible_events(request.user), form)
    alerts = visible_alerts(request.user).filter(event__in=events)
    categories, quality = stats.by_category(events), stats.quality_distribution(events)
    return render(request, "dashboard/analytics.html", {
        "form": form,
        "filters_active": len(form.active) if request.GET else 0,
        "totals": stats.overview(events, alerts),
        "categories": categories,
        "category_chart": columns(categories),
        "critical": stats.critical_frequency(events, alerts),
        "confidence": stats.confidence_histogram(events),
        "quality": quality,
        "quality_chart": columns(quality),
        "disagreement": stats.disagreement_by_class(events),
        "errors": stats.errors_by_class(events, alerts),
        "response": stats.alert_response(alerts),
        "trend": stats.daily(events, 30),
        "filter_fields": [form[name] for name in ANALYTICS_FILTERS if name in form.fields],
    })
