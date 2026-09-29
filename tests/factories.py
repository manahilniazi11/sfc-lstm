"""Small helpers that create users and events directly in the database (no models are run)."""

from datetime import timedelta

from django.utils import timezone

from sonic.detection import decision as D
from sonic.web.accounts import roles
from sonic.web.accounts.models import User
from sonic.web.alerts.models import Alert, AlertAction
from sonic.web.events.models import AudioRecord, SoundEvent

PASSWORD = "Sonic-test-2026"


def make_user(username, role=roles.USER):
    return User.objects.create_user(username=username, email=f"{username}@example.com", password=PASSWORD, role=role)


def login(client, username, role=roles.USER):
    user = make_user(username, role)
    client.login(username=username, password=PASSWORD)
    return user


def make_event(owner, final_class="Gunshot", severity="Critical", *, confidence=0.8, gtm_class=None,
               quality="Good", source=AudioRecord.Source.UPLOAD, filename="clip.wav", days_ago=0,
               review_required=False, reasons=(), status=D.CLASSIFIED, consistency="Acceptable Match",
               alert_status=D.NO_ALERT, level="High", reviewed_class="", reviewed_severity=""):
    record = AudioRecord.objects.create(owner=owner, source=source, original_filename=filename, format="WAV",
                                        file_size=100, duration_s=2, sample_rate=16000, channels=1,
                                        sha256="0" * 64, quality_grade=quality)
    gtm_class = final_class if gtm_class is None else gtm_class
    event = SoundEvent.objects.create(
        record=record, python_model_version="test", python_scores={final_class: confidence},
        python_class=final_class, python_confidence=confidence, python_margin=0.5, gtm_class=gtm_class,
        gtm_scores={gtm_class: 0.7} if gtm_class else None, gtm_confidence=0.7 if gtm_class else None,
        class_match=gtm_class == final_class, consistency=consistency, final_class=final_class,
        confidence_level=level, severity=severity, alert_status=alert_status, review_required=review_required,
        review_reasons=list(reasons), status=status, recommended_action="Check it.", reviewed_class=reviewed_class,
        reviewed_severity=reviewed_severity,
    )
    if days_ago:
        moment = timezone.now() - timedelta(days=days_ago)
        SoundEvent.objects.filter(pk=event.pk).update(created_at=moment)
        AudioRecord.objects.filter(pk=record.pk).update(uploaded_at=moment)
        event.refresh_from_db()
    return event


def make_alert(event, minutes=2, created_ago=0, status=Alert.Status.ACTIVE):
    alert = Alert.objects.create(event=event, category=event.final_class, severity=event.severity,
                                 alert_type="security", critical=True, message=f"{event.final_class} detected",
                                 recommended_action=event.recommended_action, escalate_after_minutes=minutes,
                                 status=status)
    AlertAction.objects.create(alert=alert, kind=AlertAction.Kind.CREATED, note=alert.message)
    if created_ago:
        Alert.objects.filter(pk=alert.pk).update(created_at=timezone.now() - timedelta(minutes=created_ago))
        alert.refresh_from_db()
    return alert
