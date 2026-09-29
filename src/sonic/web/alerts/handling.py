"""What people can do with an alert, and automatic escalation (FR lv, lvi; alert rule "escalation condition").

An alert moves through these statuses:

    Active ──acknowledge──► Acknowledged ──resolve──► Resolved
      │  └──escalate──► Escalated ──acknowledge──┘ │
      └──────────── dismiss (false alarm) ─────────┴──► Dismissed

- Acknowledge: someone has taken it on.
- Escalate: it needs a higher level (by a person, or automatically when an
  Active alert is not acknowledged within the rule's `escalate_after_minutes`).
- Dismiss: it was a false alarm; a reason is required.
- Resolve: the situation has been dealt with.

Every change is kept as an AlertAction (the alert's history) and in the audit trail.
"""

from __future__ import annotations

import time
from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from ..accounts.audit import Action, record as audit
from .models import Alert, AlertAction

S, K = Alert.Status, AlertAction.Kind

# action -> (statuses it can start from, status afterwards, action kind)
TRANSITIONS = {
    "acknowledge": ({S.ACTIVE, S.ESCALATED}, S.ACKNOWLEDGED, K.ACKNOWLEDGED),
    "escalate": ({S.ACTIVE, S.ACKNOWLEDGED}, S.ESCALATED, K.ESCALATED),
    "dismiss": ({S.ACTIVE, S.ACKNOWLEDGED, S.ESCALATED}, S.DISMISSED, K.DISMISSED),
    "resolve": ({S.ACKNOWLEDGED, S.ESCALATED}, S.RESOLVED, K.RESOLVED),
}
NOTE_REQUIRED = {"dismiss"}
CLOSED = {S.DISMISSED, S.RESOLVED}


class ActionRefused(Exception):
    pass


def available_actions(alert: Alert) -> list[str]:
    return [name for name, (sources, _, _) in TRANSITIONS.items() if alert.status in sources]


def act(alert: Alert, action: str, user, note: str = "", request=None) -> AlertAction:
    """Apply a person's action (or add a note with action="note")."""
    note = note.strip()
    if action == "note":
        if not note:
            raise ActionRefused("Write a note first.")
        entry = AlertAction.objects.create(alert=alert, user=user, kind=K.NOTE, note=note)
        audit(Action.ALERT_ACTION, request, user=user, target=alert, step="note")
        return entry
    if action not in TRANSITIONS:
        raise ActionRefused(f"Unknown action {action!r}.")
    sources, target, kind = TRANSITIONS[action]
    if alert.status not in sources:
        raise ActionRefused(f"A {alert.get_status_display().lower()} alert cannot be {kind.label.lower()}.")
    if action in NOTE_REQUIRED and not note:
        raise ActionRefused("Say why the alert is dismissed (for example: what the sound really was).")
    with transaction.atomic():
        previous = alert.status
        alert.status = target
        alert.handled_by, alert.handled_at = user, timezone.now()
        alert.save(update_fields=["status", "handled_by", "handled_at"])
        entry = AlertAction.objects.create(alert=alert, user=user, kind=kind, note=note)
        audit(Action.ALERT_ACTION, request, user=user, target=alert, step=action, previous=previous, status=target)
    return entry


def overdue(now=None):
    """Active alerts whose rule's escalation time has passed without anyone acknowledging them."""
    now = now or timezone.now()
    for alert in Alert.objects.filter(status=S.ACTIVE, escalate_after_minutes__isnull=False):
        if alert.created_at + timedelta(minutes=alert.escalate_after_minutes) <= now:
            yield alert


def escalate_overdue(now=None) -> list[Alert]:
    escalated = []
    for alert in overdue(now):
        with transaction.atomic():
            alert.status = S.ESCALATED
            alert.save(update_fields=["status"])
            minutes = f"{alert.escalate_after_minutes:g}"
            AlertAction.objects.create(alert=alert, kind=K.AUTO_ESCALATED,
                                       note=f"Not acknowledged within {minutes} minute{'s' if minutes != '1' else ''}.")
            audit(Action.ALERT_ACTION, None, target=alert, step="auto_escalate", status=S.ESCALATED)
        escalated.append(alert)
    return escalated


_last_check = 0.0
CHECK_EVERY_S = 30


def escalate_overdue_now_and_then() -> None:
    """Called while people use the app; the `escalate_alerts` command does the same from a scheduler."""
    global _last_check
    if time.monotonic() - _last_check >= CHECK_EVERY_S:
        _last_check = time.monotonic()
        escalate_overdue()
