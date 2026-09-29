"""Live microphone monitoring on the server (SRS Step 12, FR vi-vii, xl, lxiv, lxxix).

The browser records continuous windows (2 s by default), runs GTM on each
window itself and sends the window's audio with GTM's scores here. Each
window is validated, preprocessed, analysed by the Python model, compared
with GTM and checked against the alert rules, exactly like an upload.

Differences from uploads:
- silent windows are counted but not stored;
- repeated detection counts earlier windows of the same session (a gunshot
  heard in two windows within the configured period is confirmed; a different
  sound heard in between breaks the chain);
- a sound that continues over many windows raises one alert, not one per window;
- audio is kept only for windows that raise an alert or need review (privacy
  and storage), and only if the administrator allows it.
"""

from __future__ import annotations

import hashlib
import io
from datetime import timedelta
from pathlib import Path

import soundfile as sf
from django.conf import settings
from django.db import transaction
from django.urls import reverse
from django.utils import timezone

from sonic.detection import decision as D
from sonic.detection.compare import Scores
from sonic.inference import visuals
from sonic.inference.python_model import get_classifier
from sonic.quality.analyze import assess

from ..accounts.audit import Action, record as audit
from ..alerts.anomalies import report_model_failure
from ..alerts.models import Alert, AlertAction, DecisionSettings, current_ruleset
from .models import AudioRecord, MonitoringSession, SoundEvent
from .pipeline import MODEL_FAILURE_MESSAGE, GtmResult, _create_event, _event_marks, raise_alert

MAX_WINDOW_SECONDS = 5.0
MAX_WINDOW_BYTES = 2 * 1024 * 1024
ALERT_EPISODE_SECONDS = 60  # a new alert for the same sound only after this long without one


class WindowRejected(Exception):
    pass


def _decode_window(data: bytes):
    if not data or len(data) > MAX_WINDOW_BYTES:
        raise WindowRejected("window audio missing or too large")
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise WindowRejected(f"window audio could not be decoded ({exc.__class__.__name__})") from exc
    duration = audio.shape[0] / sr
    if duration > MAX_WINDOW_SECONDS or duration < 0.3:
        raise WindowRejected(f"window length {duration:.2f} s is outside 0.3-{MAX_WINDOW_SECONDS:.0f} s")
    return audio, sr, duration


def repeated_in_session(session: MonitoringSession, category: str, period_s: float) -> int:
    """Earlier windows of this session, within the period, that heard `category` again.

    Counting goes back from the newest window and stops at a window that heard
    a different sound: Machinery Fault, Normal Machinery, Machinery Fault is not
    a repeated fault. Background Noise, Unknown and silent windows (not stored)
    do not interrupt it, so two gunshots with quiet between them still count.
    """
    since = timezone.now() - timedelta(seconds=period_s)
    classes = (SoundEvent.objects.filter(record__session=session, created_at__gte=since)
               .order_by("-created_at").values_list("final_class", flat=True))
    count = 0
    for heard in classes:
        if heard == category:
            count += 1
        elif heard not in (D.BACKGROUND, D.UNKNOWN):
            break
    return count


def _ongoing_alert(session: MonitoringSession, category: str) -> Alert | None:
    since = timezone.now() - timedelta(seconds=ALERT_EPISODE_SECONDS)
    return (Alert.objects.filter(event__record__session=session, category=category, status__in=Alert.UNRESOLVED,
                                 created_at__gte=since).order_by("-created_at").first())


def analyse_window(session: MonitoringSession, data: bytes, gtm: GtmResult, start_s: float, request=None) -> dict:
    """Analyse one live window; returns what the live dashboard shows."""
    audio, sr, duration = _decode_window(data)
    session.windows_analysed += 1
    session.save(update_fields=["windows_analysed"])
    quality = assess(audio, sr)
    base = {"start": round(start_s, 2), "end": round(start_s + duration, 2),
            "level_dbfs": quality.metrics.get("active_level_dbfs") or quality.metrics.get("peak_dbfs")}
    if not quality.usable:
        return {**base, "silent": True, "quality": quality.grade, "issues": quality.issues}

    try:
        python = get_classifier().analyse(audio, sr)
    except Exception as exc:  # reported to administrators; the dashboard shows the message and keeps listening
        report_model_failure("Python model", exc, f"live session {session.code}")
        raise WindowRejected(MODEL_FAILURE_MESSAGE) from exc
    if python.clip_probs is None:
        return {**base, "silent": True, "quality": quality.grade, "issues": ["every window below the silence level"]}

    ruleset = current_ruleset()
    window_scores = python.window_scores()
    clip = Scores(python.clip_scores())
    earlier = repeated_in_session(session, clip.top, ruleset.thresholds.repeat_window_seconds)
    decision = D.decide(clip, Scores(gtm.scores) if gtm.scores else None, quality.grade, ruleset,
                        windows=[Scores(w) for w in window_scores], repeated_detections=earlier + 1,
                        noise_level_dbfs=quality.metrics.get("active_level_dbfs"))

    keep_audio = DecisionSettings.load().store_live_audio and (decision.alert or decision.review_required)
    with transaction.atomic():
        record = AudioRecord.objects.create(
            owner=session.user, source=AudioRecord.Source.LIVE, session=session,
            original_filename=f"{session.code} {start_s:.1f}-{start_s + duration:.1f}s.wav", format="WAV",
            file_size=len(data), duration_s=round(duration, 3), sample_rate=sr, channels=audio.shape[1], bit_depth=16,
            sha256=hashlib.sha256(data).hexdigest(), quality_grade=quality.grade, quality_issues=quality.issues,
            quality_metrics=quality.metrics,
        )
        if keep_audio:
            folder = Path(settings.UPLOAD_DIR) / record.code
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "original.wav").write_bytes(data)
            record.stored_file = f"{record.code}/original.wav"
            record.save(update_fields=["stored_file"])
            marks = _event_marks(python.window_times, window_scores, ruleset.thresholds.min_confidence)
            (folder / "waveform.png").write_bytes(visuals.waveform_png(audio, sr, marks))
            (folder / "spectrogram.png").write_bytes(visuals.spectrogram_png(audio, sr))
        event = _create_event(record, decision, python, window_scores, gtm, start_s, start_s + duration)

        alert = None
        if decision.alert:
            alert = _ongoing_alert(session, decision.final_class)
            if alert is None:
                alert = raise_alert(event, ruleset.for_category(decision.final_class), request=request, user=session.user)
            else:  # the same sound is still going on: note it on the existing alert
                AlertAction.objects.create(alert=alert, kind=AlertAction.Kind.NOTE,
                                           note=f"Detected again in {event.code} ({event.python_confidence:.0%})")
        if decision.alert or decision.review_required:
            audit(Action.PREDICTION, request, user=session.user, target=event, python=event.python_class,
                  gtm=event.gtm_class, final=event.final_class, live_session=session.code)

    return {
        **base,
        "silent": False,
        "event": event.code,
        "url": reverse("events:detail", args=[event.code]),
        "python": {"class": event.python_class, "confidence": event.python_confidence, "top3": event.ranked("python", 3)},
        "gtm": {"class": event.gtm_class, "confidence": event.gtm_confidence, "top3": event.ranked("gtm", 3), "error": event.gtm_error},
        "consistency": event.consistency,
        "confidence_difference": event.confidence_difference,
        "final_class": event.final_class,
        "confidence_level": event.confidence_level,
        "severity": event.severity,
        "alert_status": event.alert_status,
        "alert": {"code": alert.code, "category": alert.category, "severity": alert.severity, "message": alert.message,
                  "action": alert.recommended_action,
                  "url": reverse("alerts:detail", args=[alert.code])} if alert else None,
        "review_required": event.review_required,
        "review_reasons": event.review_reasons,
        "repeated": event.repeated_detections,
        "quality": quality.grade,
        "status": event.status,
        "processing_ms": event.processing_ms,
        "stored_audio": bool(record.stored_file),
    }


def start_session(user, window_seconds: float, device_label: str, request=None) -> MonitoringSession:
    session = MonitoringSession.objects.create(user=user, window_seconds=window_seconds, device_label=device_label[:200])
    audit(Action.MIC_SESSION, request, user=user, target=session, state="started", device=device_label[:200],
          window_seconds=window_seconds)
    return session


def stop_session(session: MonitoringSession, request=None) -> None:
    if session.ended_at is None:
        session.ended_at = timezone.now()
        session.save(update_fields=["ended_at"])
        audit(Action.MIC_SESSION, request, user=session.user, target=session, state="stopped",
              windows=session.windows_analysed)
