"""Monitoring and anomaly notices (FR lxxviii) and understandable errors (FR lxxvii)."""

import io
import json
from datetime import timedelta

import numpy as np
import pytest
import soundfile as sf
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.db import DatabaseError
from django.urls import reverse
from django.utils import timezone

from factories import login, make_alert, make_event, make_user
from sonic.web.accounts import roles
from sonic.web.accounts.audit import Action, record as audit
from sonic.web.accounts.models import AuditLog
from sonic.web.alerts import anomalies
from sonic.web.alerts.models import AdminNotice
from sonic.web.events import live, pipeline
from sonic.web.events.models import MonitoringSession, SoundEvent

K = AdminNotice.Kind


@pytest.fixture(autouse=True)
def upload_dir(settings, tmp_path):
    settings.UPLOAD_DIR = tmp_path / "uploads"


def kinds():
    return sorted(AdminNotice.objects.values_list("kind", flat=True))


@pytest.mark.django_db
def test_repeated_failed_uploads():
    user = make_user("mallory")
    for _ in range(4):
        audit(Action.UPLOAD_REJECTED, user=user, reason="bad file")
    assert anomalies.check() == []
    audit(Action.UPLOAD_REJECTED, user=user, reason="bad file")
    [notice] = anomalies.check()
    assert notice.kind == K.FAILED_UPLOADS and notice.count == 5 and "mallory" in notice.message
    audit(Action.UPLOAD_REJECTED, user=user, reason="bad file")
    anomalies.check()
    assert AdminNotice.objects.count() == 1 and AdminNotice.objects.get().count == 6  # updated, not repeated


@pytest.mark.django_db
def test_acknowledged_notice_only_returns_for_new_occurrences():
    for _ in range(5):
        audit(Action.LOGIN_FAILED, username="admin")
    anomalies.check()
    AdminNotice.objects.update(acknowledged_at=timezone.now())
    anomalies.check()
    assert AdminNotice.objects.count() == 1  # the same five failures do not raise it again
    for _ in range(5):
        audit(Action.LOGIN_FAILED, username="admin")
    anomalies.check()
    assert AdminNotice.objects.filter(acknowledged_at__isnull=True, kind=K.FAILED_LOGINS).count() == 1


@pytest.mark.django_db
def test_failed_logins_from_one_address():
    for i in range(5):
        AuditLog.objects.create(action=Action.LOGIN_FAILED, username=f"guess{i}", ip_address="10.0.0.9")
    [notice] = anomalies.check()
    assert notice.kind == K.FAILED_LOGINS and notice.key == "ip_address:10.0.0.9"


@pytest.mark.django_db
def test_duplicate_files():
    owner = make_user("owner")
    original = make_event(owner).record
    for _ in range(3):
        copy = make_event(owner).record
        copy.duplicate_of = original
        copy.save()
    [notice] = anomalies.check()
    assert notice.kind == K.DUPLICATES and notice.count == 3


@pytest.mark.django_db
def test_excessive_critical_alerts():
    owner = make_user("owner")
    for _ in range(5):
        make_alert(make_event(owner))
    assert kinds() == [] and [n.kind for n in anomalies.check()] == [K.CRITICAL_ALERTS]


@pytest.mark.django_db
def test_low_confidence_spike():
    owner = make_user("owner")
    for i in range(20):  # the usual week: 10% Low
        make_event(owner, level="Low" if i < 2 else "High", days_ago=2)
    for i in range(10):  # the last hour: 60% Low
        make_event(owner, level="Low" if i < 6 else "High")
    [notice] = anomalies.check()
    assert notice.kind == K.LOW_CONFIDENCE and "60%" in notice.message


@pytest.mark.django_db
def test_no_spike_when_low_confidence_is_usual():
    owner = make_user("owner")
    for i in range(20):
        make_event(owner, level="Low" if i < 12 else "High", days_ago=2)
    for i in range(10):
        make_event(owner, level="Low" if i < 6 else "High")
    assert anomalies.check() == []


@pytest.mark.django_db
def test_gtm_failures():
    owner = make_user("owner")
    for _ in range(5):
        SoundEvent.objects.filter(pk=make_event(owner).pk).update(gtm_error="WebGL not available")
    [notice] = anomalies.check()
    assert notice.kind == K.MODEL_FAILURE and notice.key == "GTM"


def wav(seconds=2.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    buffer = io.BytesIO()
    sf.write(buffer, (0.4 * np.sin(2 * np.pi * 800 * t)).astype(np.float32), sr, format="WAV")
    return buffer.getvalue()


class Broken:
    version, classes = "broken", []

    def analyse(self, audio, sr):
        raise RuntimeError("weights file missing")


@pytest.mark.django_db
def test_model_failure_on_upload_is_a_message_and_a_notice(client, monkeypatch):
    monkeypatch.setattr(pipeline, "get_classifier", lambda: Broken())
    login(client, "alice")
    response = client.post(reverse("events:analyse"), {"audio": SimpleUploadedFile("a.wav", wav())},
                           HTTP_X_REQUESTED_WITH="fetch")
    assert response.status_code == 400 and "administrators have been notified" in response.json()["error"]
    notice = AdminNotice.objects.get(kind=K.MODEL_FAILURE)
    assert "weights file missing" in notice.message and not SoundEvent.objects.exists()


@pytest.mark.django_db
def test_model_failure_in_live_monitoring(client, monkeypatch):
    monkeypatch.setattr(live, "get_classifier", lambda: Broken())
    user = login(client, "alice")
    session = MonitoringSession.objects.create(user=user)
    response = client.post(reverse("monitoring:window", args=[session.code]),
                           {"audio": SimpleUploadedFile("w.wav", wav()), "start": "0"})
    assert response.status_code == 400 and "administrators have been notified" in response.json()["error"]
    assert AdminNotice.objects.filter(kind=K.MODEL_FAILURE, key="Python model").exists()


@pytest.mark.django_db
def test_database_failure_page(client, monkeypatch):
    login(client, "alice")
    from sonic.web.events import views

    def fail(*args, **kwargs):
        raise DatabaseError("database is locked")

    monkeypatch.setattr(views, "visible_events", fail)
    response = client.get(reverse("events:history"))
    assert response.status_code == 503 and b"database is not available" in response.content


@pytest.mark.django_db
def test_notices_page_and_acknowledge(client):
    anomalies.notify(K.DUPLICATES, "bob", "bob uploaded 3 duplicate files.", 3)
    login(client, "rev", roles.REVIEWER)
    assert client.get(reverse("settings:notices")).status_code == 403
    login(client, "admin1", roles.ADMIN)
    home = client.get(reverse("home"))
    assert home.context["open_notices"] == 1 and b"bob uploaded 3 duplicate files." in home.content
    client.post(reverse("settings:acknowledge", args=[AdminNotice.objects.get().pk]))
    assert AdminNotice.objects.get().acknowledged_by.username == "admin1"
    assert client.get(reverse("settings:notices")).status_code == 200


@pytest.mark.django_db
def test_check_command(capsys):
    for _ in range(5):
        audit(Action.LOGIN_FAILED, username="root")
    call_command("check_anomalies")
    assert "Failed login attempts" in capsys.readouterr().out
