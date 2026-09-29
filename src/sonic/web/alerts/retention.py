"""Data retention (FR lxxx): delete stored audio and event records older than the administrator's limits.

Two separate limits (Decision settings):
- `audio_retention_days`: after this, the audio file and its images are
  deleted; the event record (predictions, scores, decision) stays.
- `event_retention_days`: after this, the whole record is deleted (audio
  metadata, event, its alerts and review history).

Records that still need a person are kept whatever their age: an event
waiting for review, or with an alert nobody has closed. Run it from the
Settings page or with `python manage.py apply_retention` (add --dry-run to
only count).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from sonic.detection import decision as D

from ..accounts.audit import Action, record as audit
from .models import Alert, DecisionSettings


@dataclass
class RetentionPlan:
    audio_before: object
    events_before: object
    audio_files: int
    event_records: int
    kept_open: int  # old records kept because they still need a person


def _still_needed() -> Q:
    return (Q(event__review_required=True) & ~Q(event__status__in=[D.REVIEWED, D.CLOSED])) | \
        Q(event__alerts__status__in=Alert.UNRESOLVED)


def _old_records(before):
    from ..events.models import AudioRecord

    return AudioRecord.objects.filter(uploaded_at__lt=before)


def plan(now=None) -> RetentionPlan:
    now = now or timezone.now()
    config = DecisionSettings.load()
    audio_before = now - timedelta(days=config.audio_retention_days)
    events_before = now - timedelta(days=config.event_retention_days)
    old_events = _old_records(events_before)
    return RetentionPlan(
        audio_before=audio_before,
        events_before=events_before,
        # records past the event limit are counted once, as event records (their audio goes with them)
        audio_files=(_old_records(audio_before).filter(uploaded_at__gte=events_before).exclude(stored_file="")
                     .exclude(_still_needed()).distinct().count()),
        event_records=old_events.exclude(_still_needed()).distinct().count(),
        kept_open=old_events.filter(_still_needed()).distinct().count(),
    )


def _delete_folder(record) -> None:
    """Delete the record's own folder (UPLOAD_DIR/AUD-xxxxxx/): never the upload directory itself or anything outside it."""
    if not record.stored_file:
        return
    root = Path(settings.UPLOAD_DIR).resolve()
    folder = (root / record.stored_file).parent.resolve()
    if folder.is_dir() and folder != root and folder.is_relative_to(root):
        shutil.rmtree(folder)


def apply(user=None, request=None, now=None) -> RetentionPlan:
    todo = plan(now)
    with transaction.atomic():
        old_events = list(_old_records(todo.events_before).exclude(_still_needed()).distinct())
        for record in old_events:
            _delete_folder(record)
            record.delete()  # the event, its alerts and reviews go with it
        old_audio = list(_old_records(todo.audio_before).exclude(stored_file="").exclude(_still_needed()).distinct())
        for record in old_audio:
            _delete_folder(record)
            record.stored_file = ""
            record.save(update_fields=["stored_file"])
        audit(Action.RETENTION, request, user=user, audio_files=len(old_audio), event_records=len(old_events),
              kept_open=todo.kept_open)
    return RetentionPlan(todo.audio_before, todo.events_before, len(old_audio), len(old_events), todo.kept_open)
