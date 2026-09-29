"""Monitoring and anomaly notices for administrators (FR lxxviii).

`check()` looks at the recent audit trail, events and alerts and raises an
AdminNotice when something unusual happens:

| Notice                        | Raised when (within the period)                                   |
|-------------------------------|-------------------------------------------------------------------|
| Repeated failed uploads       | one user has 5+ rejected uploads in 15 minutes                    |
| Failed login attempts         | 5+ failed logins for one username, or from one IP, in 15 minutes  |
| Duplicate files               | one user uploads 3+ exact or near-duplicate files in 60 minutes   |
| Excessive critical alerts     | 5+ critical alerts are raised in 60 minutes                       |
| Unusual low-confidence spike  | in the last hour at least 10 results, half or more of them Low    |
|                               | confidence, and at least twice the share of the previous 7 days   |
| Model failure                 | the Python model raises an error (reported at once by `report_model_failure`), |
|                               | or GTM reports an error for 5+ results in 60 minutes              |

While a notice is open, new occurrences update it (count, last seen)
instead of adding more. After an administrator acknowledges it, only
occurrences after that moment can raise it again.

`check()` runs at most once a minute while administrators use the app, and
from `python manage.py check_anomalies` for a scheduler.
"""

from __future__ import annotations

import time
from datetime import timedelta

from django.db.models import Count, Q
from django.utils import timezone

from ..accounts.models import AuditLog
from .models import AdminNotice, Alert

K = AdminNotice.Kind

FAILED_UPLOADS = (5, 15)  # (how many, within minutes)
FAILED_LOGINS = (5, 15)
DUPLICATES = (3, 60)
CRITICAL_ALERTS = (5, 60)
GTM_FAILURES = (5, 60)
LOW_CONFIDENCE_MIN_RESULTS = 10
LOW_CONFIDENCE_SHARE = 0.5


def _since(kind: str, key: str, minutes: float, now):
    """Start of the counting period: never before the last time this notice was acknowledged."""
    start = now - timedelta(minutes=minutes)
    acknowledged = (AdminNotice.objects.filter(kind=kind, key=key, acknowledged_at__isnull=False)
                    .order_by("-acknowledged_at").values_list("acknowledged_at", flat=True).first())
    return max(start, acknowledged) if acknowledged else start


def notify(kind: str, key: str, message: str, count: int = 1, now=None) -> AdminNotice:
    """Raise a notice, or update the open one about the same thing."""
    now = now or timezone.now()
    notice = AdminNotice.objects.filter(kind=kind, key=key, acknowledged_at__isnull=True).first()
    if notice:
        notice.message, notice.count, notice.last_seen = message[:300], max(count, notice.count), now
        notice.save(update_fields=["message", "count", "last_seen"])
        return notice
    notice = AdminNotice.objects.create(kind=kind, key=key, message=message[:300], count=count)
    AdminNotice.objects.filter(pk=notice.pk).update(first_seen=now, last_seen=now)
    return notice


def report_model_failure(model: str, error: Exception, context: str) -> AdminNotice:
    """Called where the Python model is run, when it raises."""
    open_notice = AdminNotice.objects.filter(kind=K.MODEL_FAILURE, key=model, acknowledged_at__isnull=True).first()
    count = open_notice.count + 1 if open_notice else 1
    return notify(K.MODEL_FAILURE, model, f"The {model} failed while analysing {context}: "
                                          f"{error.__class__.__name__}: {error}", count)


def _audit_groups(action: str, field: str, minutes: float, now):
    rows = AuditLog.objects.filter(action=action, created_at__gte=now - timedelta(minutes=minutes))
    rows = rows.filter(ip_address__isnull=False) if field == "ip_address" else rows.exclude(**{field: ""})
    return rows.values(field).annotate(n=Count("id"))


def _check_audit(kind, action, field, limit, describe, now):
    raised = []
    threshold, minutes = limit
    for group in _audit_groups(action, field, minutes, now):
        key = f"{field}:{group[field]}"
        since = _since(kind, key, minutes, now)
        n = AuditLog.objects.filter(action=action, created_at__gte=since, **{field: group[field]}).count()
        if n >= threshold:
            raised.append(notify(kind, key, describe(group[field], n, minutes), n, now))
    return raised


def check(now=None) -> list[AdminNotice]:
    from ..events.models import AudioRecord, SoundEvent

    now = now or timezone.now()
    raised = []
    raised += _check_audit(K.FAILED_UPLOADS, AuditLog.Action.UPLOAD_REJECTED, "username", FAILED_UPLOADS,
                           lambda who, n, m: f"{who} had {n} uploads rejected within {m} minutes.", now)
    for field in ["username", "ip_address"]:
        raised += _check_audit(K.FAILED_LOGINS, AuditLog.Action.LOGIN_FAILED, field, FAILED_LOGINS,
                               lambda who, n, m: f"{n} failed logins for {who} within {m} minutes.", now)

    threshold, minutes = DUPLICATES
    duplicates = (AudioRecord.objects.filter(uploaded_at__gte=now - timedelta(minutes=minutes))
                  .filter(Q(duplicate_of__isnull=False) | Q(near_duplicate_of__isnull=False))
                  .values("owner__username").annotate(n=Count("id")))
    for group in duplicates:
        who = group["owner__username"]
        since = _since(K.DUPLICATES, who, minutes, now)
        n = (AudioRecord.objects.filter(uploaded_at__gte=since, owner__username=who)
             .filter(Q(duplicate_of__isnull=False) | Q(near_duplicate_of__isnull=False)).count())
        if n >= threshold:
            raised.append(notify(K.DUPLICATES, who, f"{who} uploaded {n} duplicate or near-duplicate files "
                                                    f"within {minutes} minutes.", n, now))

    threshold, minutes = CRITICAL_ALERTS
    n = Alert.objects.filter(critical=True, created_at__gte=_since(K.CRITICAL_ALERTS, "", minutes, now)).count()
    if n >= threshold:
        raised.append(notify(K.CRITICAL_ALERTS, "", f"{n} critical alerts were raised within {minutes} minutes.", n, now))

    threshold, minutes = GTM_FAILURES
    n = SoundEvent.objects.filter(created_at__gte=_since(K.MODEL_FAILURE, "GTM", minutes, now)).exclude(gtm_error="").count()
    if n >= threshold:
        raised.append(notify(K.MODEL_FAILURE, "GTM", f"GTM gave no result for {n} recordings within {minutes} "
                                                     f"minutes (the browser could not run the model).", n, now))

    hour = SoundEvent.objects.filter(created_at__gte=_since(K.LOW_CONFIDENCE, "", 60, now))
    week = SoundEvent.objects.filter(created_at__gte=now - timedelta(days=7), created_at__lt=now - timedelta(hours=1))
    recent = hour.aggregate(n=Count("id"), low=Count("id", filter=Q(confidence_level="Low")))
    before = week.aggregate(n=Count("id"), low=Count("id", filter=Q(confidence_level="Low")))
    if recent["n"] >= LOW_CONFIDENCE_MIN_RESULTS:
        share = recent["low"] / recent["n"]
        usual = before["low"] / before["n"] if before["n"] else 0.0
        if share >= LOW_CONFIDENCE_SHARE and share >= 2 * usual:
            raised.append(notify(K.LOW_CONFIDENCE, "", f"{share:.0%} of the last hour's {recent['n']} results were "
                                                       f"Low confidence (usually {usual:.0%}): check microphones, "
                                                       f"audio quality or the models.", recent["low"], now))
    return raised


_last_check = 0.0
CHECK_EVERY_S = 60


def check_now_and_then() -> None:
    global _last_check
    if time.monotonic() - _last_check >= CHECK_EVERY_S:
        _last_check = time.monotonic()
        check()
