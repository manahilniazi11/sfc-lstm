"""Live monitoring: sessions, windows, silence, repeated detection and alert de-duplication (microphone tests)."""

import io
import json

import numpy as np
import pytest
import soundfile as sf
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from sonic.detection.compare import Scores
from sonic.web.accounts.models import AuditLog, User
from sonic.web.alerts.models import Alert, AlertAction
from sonic.web.events import live
from sonic.web.events.models import AudioRecord, MonitoringSession, SoundEvent

PASSWORD = "Sonic-test-2026"
SR = 48000


@pytest.fixture(autouse=True)
def upload_dir(settings, tmp_path):
    settings.UPLOAD_DIR = tmp_path / "uploads"


def login(client, username="alice"):
    user = User.objects.create_user(username=username, email=f"{username}@example.com", password=PASSWORD)
    client.login(username=username, password=PASSWORD)
    return user


def window_wav(y):
    buffer = io.BytesIO()
    sf.write(buffer, y, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def siren(seconds=2.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.4 * np.sin(2 * np.pi * (700 + 300 * np.sin(2 * np.pi * 0.8 * t)) * t)).astype(np.float32)


def start(client):
    response = client.post(reverse("monitoring:start"), {"device": "Stereo Mix (Realtek)"})
    assert response.status_code == 200
    return response.json()


def send(client, session, y, start_s=0.0, gtm=None):
    return client.post(session["window_url"], {
        "audio": SimpleUploadedFile("window.wav", window_wav(y)),
        "gtm": gtm or json.dumps({"version": "gtm-test", "error": "not run"}),
        "start": str(start_s),
    })


@pytest.mark.django_db
def test_page_needs_login_and_renders(client):
    assert client.get(reverse("monitoring:monitor")).status_code == 302
    login(client)
    page = client.get(reverse("monitoring:monitor"))
    assert page.status_code == 200 and b"Nothing is captured until you press Start" in page.content


@pytest.mark.django_db
def test_session_lifecycle_is_audited(client):
    login(client)
    session = start(client)
    assert MonitoringSession.objects.get(code=session["session"]).active
    client.post(session["stop_url"])
    stored = MonitoringSession.objects.get(code=session["session"])
    assert not stored.active and stored.device_label.startswith("Stereo Mix")
    states = [log.details["state"] for log in AuditLog.objects.filter(action=AuditLog.Action.MIC_SESSION).order_by("created_at")]
    assert states == ["started", "stopped"]
    assert send(client, session, siren()).status_code == 409  # windows after Stop are refused


@pytest.mark.django_db
def test_a_sound_window_is_analysed_and_stored_as_an_event(client):
    login(client)
    session = start(client)
    result = send(client, session, siren(), start_s=4.0).json()
    assert not result["silent"] and result["event"].startswith("EVT-")
    assert result["start"] == 4.0 and result["end"] == pytest.approx(6.0, abs=0.01)
    assert result["python"]["class"] and len(result["python"]["top3"]) == 3
    event = SoundEvent.objects.get(code=result["event"])
    assert event.record.source == "live" and event.record.session.code == session["session"]
    assert event.segment_start_s == 4.0


@pytest.mark.django_db
def test_silent_windows_are_counted_but_not_stored(client):
    login(client)
    session = start(client)
    result = send(client, session, np.zeros(2 * SR, np.float32)).json()
    assert result["silent"] and not SoundEvent.objects.exists()
    assert MonitoringSession.objects.get(code=session["session"]).windows_analysed == 1


@pytest.mark.django_db
def test_bad_windows_are_rejected(client):
    login(client)
    session = start(client)
    assert send(client, session, siren(8.0)).status_code == 400  # longer than a live window
    response = client.post(session["window_url"], {"audio": SimpleUploadedFile("w.wav", b"junk"), "start": "0"})
    assert response.status_code == 400


@pytest.mark.django_db
def test_other_users_cannot_use_a_session(client):
    login(client, "alice")
    session = start(client)
    client.logout()
    login(client, "bob")
    assert send(client, session, siren()).status_code == 404


@pytest.mark.django_db
def test_repeated_detection_counts_earlier_windows_in_the_period(client):
    user = login(client)
    session = MonitoringSession.objects.create(user=user)
    assert live.repeated_in_session(session, "Gunshot", 10) == 0
    code = send(client, {"window_url": reverse("monitoring:window", args=[session.code])}, siren()).json()["event"]
    category = SoundEvent.objects.get(code=code).final_class
    assert live.repeated_in_session(session, category, 10) == 1
    assert live.repeated_in_session(session, category, 0) == 0


def heard(session, category):
    """A stored live window whose final class was `category` (only the fields the repeat count reads matter)."""
    record = AudioRecord.objects.create(owner=session.user, source=AudioRecord.Source.LIVE, session=session,
                                        original_filename="w.wav", format="WAV", file_size=1, duration_s=2, channels=1,
                                        sample_rate=SR, sha256="0" * 64, quality_grade="Good")
    SoundEvent.objects.create(record=record, python_model_version="t", python_class=category, python_confidence=0.9,
                              python_margin=0.5, consistency="-", final_class=category, confidence_level="High",
                              severity="Low", alert_status="No alert")


@pytest.mark.django_db
def test_a_different_sound_in_between_breaks_the_repeat(client):
    session = MonitoringSession.objects.create(user=login(client))
    for category in ["Machinery Fault", "Background Noise", "Machinery Fault", "Normal Machinery"]:
        heard(session, category)
    assert live.repeated_in_session(session, "Machinery Fault", 10) == 0  # Normal Machinery came last
    heard(session, "Machinery Fault")
    heard(session, "Unknown")
    heard(session, "Background Noise")
    assert live.repeated_in_session(session, "Machinery Fault", 10) == 1  # Unknown and background do not interrupt


@pytest.mark.django_db
def test_a_continuing_sound_raises_one_alert(client, monkeypatch):
    """Force every window to be a confident, agreed Glass Breaking: alerts must not repeat per window."""
    user = login(client)
    session = MonitoringSession.objects.create(user=user)
    url = {"window_url": reverse("monitoring:window", args=[session.code])}

    class Fixed:
        version, classes = "fixed", ["Glass Breaking", "Background Noise"]

        def analyse(self, audio, sr):
            from sonic.inference.python_model import PythonResult

            probs = np.array([[0.95, 0.05]] * 3)
            return PythonResult("fixed", self.classes, [(0, 1), (0.5, 1.5), (1, 2)], probs, probs[0], 5.0, {})

    monkeypatch.setattr(live, "get_classifier", lambda: Fixed())
    gtm = json.dumps({"version": "g", "labels": ["Glass Breaking", "Background Noise"], "clip": [0.9, 0.1]})
    first = send(client, url, siren(), gtm=gtm).json()
    second = send(client, url, siren(), 2.0, gtm=gtm).json()
    assert first["alert_status"] == "Alert generated" and second["alert_status"] == "Alert generated"
    assert Alert.objects.count() == 1 and first["alert"]["code"] == second["alert"]["code"]
    assert AlertAction.objects.filter(kind=AlertAction.Kind.NOTE).count() == 1
    assert first["stored_audio"]  # alert windows keep their audio for review


def test_scores_helper_sanity():
    assert Scores({"A": 0.7, "B": 0.3}).top == "A"
