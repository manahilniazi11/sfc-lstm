"""Manual review (FR lvii-lxii): the queue, and what a reviewer can decide.

A reviewer listens to the recording and then either
- **confirms** the automatic class, or
- **corrects** it to another class (a decision override).

Either way they may also override the severity and the recommended action,
add a comment, and close the event. The models' outputs (Python and GTM
predictions, scores, the automatic final class and severity) are never
changed: the reviewer's decision is stored next to them (`reviewed_*`
fields) and every step is kept as a ReviewEntry and in the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass

from django.db import transaction
from django.db.models import Case, IntegerField, Value, When
from django.utils import timezone

from sonic.detection import decision as D
from sonic.detection.rules import SEVERITIES, UNKNOWN

from ..accounts.audit import Action, record as audit
from ..alerts.models import AlertAction, current_ruleset
from .models import ReviewEntry, SoundEvent

K = ReviewEntry.Kind
DONE = [D.REVIEWED, D.CLOSED]
SEVERITY_RANK = Case(*[When(severity=s, then=Value(i)) for i, s in enumerate(reversed(SEVERITIES))],
                     default=Value(len(SEVERITIES)), output_field=IntegerField())


class ReviewRefused(Exception):
    pass


def queue():
    """Events waiting for a reviewer: most severe first, then oldest first (it has waited longest)."""
    return (SoundEvent.objects.select_related("record", "record__owner", "record__session")
            .filter(review_required=True).exclude(status__in=DONE)
            .annotate(rank=SEVERITY_RANK).order_by("rank", "created_at"))


def class_choices() -> list[str]:
    """Every class a reviewer may choose: all rule categories (including ones the models cannot predict yet)."""
    return sorted(c for c in current_ruleset().rules if c != UNKNOWN)


@dataclass
class ReviewDecision:
    decision: str  # "confirm" or "correct"
    category: str = ""
    severity: str = ""  # blank: keep the automatic one (or the corrected class's rule)
    recommended_action: str = ""
    comment: str = ""
    close: bool = False


def decide(event: SoundEvent, reviewer, choice: ReviewDecision, request=None) -> ReviewEntry:
    if choice.decision not in ("confirm", "correct"):
        raise ReviewRefused("Choose whether to confirm or correct the class.")
    if choice.severity and choice.severity not in SEVERITIES:
        raise ReviewRefused(f"Unknown severity {choice.severity!r}.")
    if choice.decision == "confirm":
        if event.final_class == UNKNOWN:
            raise ReviewRefused("The models could not classify this sound: choose the correct class instead.")
        new_class = event.final_class
    else:
        new_class = choice.category
        if new_class not in class_choices():
            raise ReviewRefused("Choose the class you heard.")
        if new_class == event.final_class:
            choice.decision = "confirm"  # "correcting" to the same class is a confirmation

    rule = current_ruleset().for_category(new_class)
    corrected = choice.decision == "correct"
    severity = choice.severity or (rule.severity if corrected else "")
    action = choice.recommended_action.strip() or (rule.recommended_action if corrected else "")
    previous = event.effective_class

    with transaction.atomic():
        event.reviewed_class = new_class
        event.reviewed_severity = severity
        event.reviewed_action = action
        event.reviewed_by, event.reviewed_at = reviewer, timezone.now()
        event.status = D.CLOSED if choice.close else D.REVIEWED
        event.save(update_fields=["reviewed_class", "reviewed_severity", "reviewed_action", "reviewed_by",
                                  "reviewed_at", "status"])
        entry = ReviewEntry.objects.create(
            event=event, reviewer=reviewer, kind=K.CORRECT if corrected else K.CONFIRM, previous_class=previous,
            new_class=new_class, severity=severity, recommended_action=choice.recommended_action.strip(),
            comment=choice.comment.strip(),
        )
        if choice.close:
            ReviewEntry.objects.create(event=event, reviewer=reviewer, kind=K.CLOSE)
        # The original outputs go into the audit trail too, so an override can always be traced back.
        audit(Action.OVERRIDE if corrected else Action.REVIEW, request, user=reviewer, target=event,
              decision=choice.decision, python=event.python_class, gtm=event.gtm_class or None,
              automatic=event.final_class, reviewed=new_class, severity=event.effective_severity,
              status=event.status)
        # Tell whoever handles the event's alerts that a person heard something else.
        if corrected:
            for alert in event.alerts.exclude(category=new_class):
                AlertAction.objects.create(alert=alert, user=reviewer, kind=AlertAction.Kind.NOTE,
                                           note=f"Reviewer corrected the sound to {new_class}.")
    return entry


def comment(event: SoundEvent, reviewer, text: str, recommended_action: str = "", request=None) -> ReviewEntry:
    text, recommended_action = text.strip(), recommended_action.strip()
    if not text and not recommended_action:
        raise ReviewRefused("Write a comment or a recommended action.")
    with transaction.atomic():
        if recommended_action:
            event.reviewed_action = recommended_action
            event.save(update_fields=["reviewed_action"])
        entry = ReviewEntry.objects.create(event=event, reviewer=reviewer, kind=K.COMMENT, comment=text,
                                           recommended_action=recommended_action)
        audit(Action.REVIEW, request, user=reviewer, target=event, decision="comment")
    return entry


def close(event: SoundEvent, reviewer, request=None) -> ReviewEntry:
    if event.status == D.CLOSED:
        raise ReviewRefused("The event is already closed.")
    with transaction.atomic():
        event.status = D.CLOSED
        event.save(update_fields=["status"])
        entry = ReviewEntry.objects.create(event=event, reviewer=reviewer, kind=K.CLOSE)
        audit(Action.REVIEW, request, user=reviewer, target=event, decision="close")
    return entry


def reopen(event: SoundEvent, reviewer, request=None) -> ReviewEntry:
    """Back into the queue, e.g. when a closed decision turns out to be wrong. Earlier decisions stay in the history."""
    if event.status not in DONE:
        raise ReviewRefused("Only reviewed or closed events can be reopened.")
    with transaction.atomic():
        event.status = D.MANUAL_REVIEW
        event.review_required = True
        event.save(update_fields=["status", "review_required"])
        entry = ReviewEntry.objects.create(event=event, reviewer=reviewer, kind=K.REOPEN)
        audit(Action.REVIEW, request, user=reviewer, target=event, decision="reopen")
    return entry
