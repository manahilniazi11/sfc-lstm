"""Template context: the menu badges (alerts needing attention, events waiting for review, admin notices)."""

from .anomalies import check_now_and_then
from .handling import escalate_overdue_now_and_then
from .models import AdminNotice, Alert


def visible_alerts(user):
    """Operators see every alert; a normal user sees alerts raised by their own recordings."""
    alerts = Alert.objects.select_related("event", "event__record", "event__record__owner")
    return alerts if user.can("view_all_events") else alerts.filter(event__record__owner=user)


def review_queue_count() -> int:
    from ..events.review import queue

    return queue().count()


def counts(request):
    user = request.user
    if not user.is_authenticated:
        return {}
    escalate_overdue_now_and_then()
    context = {
        "open_alerts": visible_alerts(user).filter(status__in=Alert.OPEN).count(),
        "review_waiting": review_queue_count() if user.can("review") else 0,
    }
    if user.can("administer"):
        check_now_and_then()
        context["open_notices"] = AdminNotice.objects.filter(acknowledged_at__isnull=True).count()
    return context
