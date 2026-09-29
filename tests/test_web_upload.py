"""Upload -> validation -> both models -> decision -> storage (integration, negative, duplicate and security tests)."""

import csv
import io
import json

import numpy as np
import pytest
import soundfile as sf
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from sonic.dataset.config import DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR
from sonic.web.accounts import roles
from sonic.web.accounts.models import AuditLog, User
from sonic.web.events import pipeline
from sonic.web.events.models import AudioRecord, SoundEvent

PASSWORD = "Sonic-test-2026"
SR = 16000
GTM_LABELS = ["Aggression", "Alarm or Siren", "Animal Sound", "Background Noise", "Glass Breaking",
              "Gunshot", "Machinery Fault", "Normal Machinery", "Panic Scream", "Vehicle Horn"]


@pytest.fixture(autouse=True)
def upload_dir(settings, tmp_path):
    settings.UPLOAD_DIR = tmp_path / "uploads"
    return settings.UPLOAD_DIR


def login(client, username="alice", role=roles.USER):
    user = User.objects.create_user(username=username, email=f"{username}@example.com", password=PASSWORD, role=role)
    client.login(username=username, password=PASSWORD)
    return user


def wav_bytes(y, sr=SR, fmt="WAV"):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format=fmt)
    return buffer.getvalue()


def siren(seconds=3.0, seed=0):
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(seed)
    y = 0.4 * np.sin(2 * np.pi * (700 + 300 * np.sin(2 * np.pi * 0.8 * t)) * t) + 0.01 * rng.standard_normal(t.size)
    return y.astype(np.float32)


def gtm_payload(top="Alarm or Siren", p=0.7):
    rest = (1 - p) / (len(GTM_LABELS) - 1)
    clip = [p if label == top else rest for label in GTM_LABELS]
    return json.dumps({"version": "gtm-test", "labels": GTM_LABELS, "clip": clip,
                       "windows": [{"start": 0, "end": 1, "probs": clip}]})


def post(client, name, data, gtm=None, batch=False):
    body = {"audio": SimpleUploadedFile(name, data), "gtm": gtm if gtm is not None else gtm_payload()}
    if batch:
        body["batch"] = "1"
    return client.post(reverse("events:analyse"), body, HTTP_X_REQUESTED_WITH="fetch")


@pytest.mark.django_db
def test_valid_upload_is_analysed_by_both_models_and_stored(client, upload_dir):
    user = login(client)
    response = post(client, "siren.wav", wav_bytes(siren()))
    assert response.status_code == 200, response.content
    result = response.json()
    assert result["ok"] and result["code"].startswith("EVT-")
    event = SoundEvent.objects.get(code=result["code"])
    record = event.record
    assert record.owner == user and record.format == "WAV" and record.sample_rate == SR and record.channels == 1
    assert record.bit_depth == 16 and record.duration_s == pytest.approx(3.0, abs=0.01)
    assert set(event.python_scores) >= {"Gunshot", "Background Noise"}
    assert sum(event.python_scores.values()) == pytest.approx(1, abs=1e-3)
    assert event.gtm_model_version == "gtm-test" and event.gtm_class == "Alarm or Siren"
    assert event.confidence_difference == pytest.approx(abs(event.python_confidence - event.gtm_confidence), abs=1e-4)
    assert event.python_model_version.startswith("ensemble")
    folder = upload_dir / record.code
    assert (folder / "original.wav").exists() and (folder / "waveform.png").exists() and (folder / "spectrogram.png").exists()
    actions = set(AuditLog.objects.values_list("action", flat=True))
    assert {AuditLog.Action.UPLOAD, AuditLog.Action.PREDICTION} <= actions
    page = client.get(reverse("events:detail", args=[event.code]))
    assert page.status_code == 200 and b"Google Teachable Machine" in page.content
    html = page.content.decode()
    assert "What happened" in html and "Confidence for every class" in html and "Preprocessing (Python model)" in html
    assert event.windows["preprocessing"]["sample_rate"] == 16000 and event.windows["preprocessing"]["windows_total"] >= 1


@pytest.mark.django_db
@pytest.mark.parametrize("name,data,message", [
    ("notes.txt", b"hello", "Unsupported format"),
    ("broken.wav", b"RIFF\x00\x00\x00\x00WAVEjunk", "could not be decoded"),
    ("silence.wav", wav_bytes(np.zeros(SR * 2, np.float32)), "unusable"),
    ("tiny.wav", wav_bytes(siren(0.1)), "unusable"),
    ("empty.wav", b"", "empty"),
], ids=["text-file", "damaged", "silent", "too-short", "empty"])
def test_invalid_uploads_are_rejected_with_a_message(client, name, data, message):
    login(client)
    response = post(client, name, data)
    assert response.status_code == 400
    assert message.lower() in response.json()["error"].lower()
    assert not AudioRecord.objects.exists()


@pytest.mark.django_db
def test_rejections_are_audited(client):
    login(client)
    post(client, "silence.wav", wav_bytes(np.zeros(SR * 2, np.float32)))
    log = AuditLog.objects.get(action=AuditLog.Action.UPLOAD_REJECTED)
    assert log.details["filename"] == "silence.wav" and "unusable" in log.details["reason"].lower()


@pytest.mark.django_db
def test_too_low_sampling_rate_is_rejected(client):
    login(client)
    response = post(client, "low.wav", wav_bytes(siren()[::4], sr=4000))
    assert response.status_code == 400 and "sampling rate" in response.json()["error"].lower()


@pytest.mark.django_db
def test_exact_and_near_duplicates_are_detected(client):
    login(client)
    y = siren(seconds=4, seed=1)
    first = post(client, "a.wav", wav_bytes(y)).json()
    again = post(client, "a-copy.wav", wav_bytes(y)).json()
    quieter = post(client, "a-quiet.flac", wav_bytes(y * 0.3, fmt="FLAC")).json()
    original = AudioRecord.objects.get(event__code=first["code"])
    assert AudioRecord.objects.get(event__code=again["code"]).duplicate_of == original
    near = AudioRecord.objects.get(event__code=quieter["code"])
    assert near.duplicate_of is None and near.near_duplicate_of == original and near.near_duplicate_score > 0.9


@pytest.mark.django_db
def test_missing_gtm_result_sends_the_event_to_review(client):
    login(client)
    result = post(client, "siren.wav", wav_bytes(siren()), gtm=json.dumps({"version": "gtm-test", "error": "decode failed"})).json()
    event = SoundEvent.objects.get(code=result["code"])
    assert event.gtm_scores is None and event.gtm_error == "decode failed"
    assert "GTM result unavailable" in event.review_reasons and event.consistency == "Uncertain Result"


@pytest.mark.parametrize("payload", [
    "not json",
    json.dumps({"labels": ["A", "B"], "clip": [0.9]}),
    json.dumps({"labels": ["A", "B"], "clip": [0.9, 0.9]}),  # does not sum to 1
    json.dumps({"labels": ["A", "A"], "clip": [0.5, 0.5]}),
    json.dumps({"labels": ["A", "B"], "clip": [float("nan"), 1]}),
])
def test_malformed_gtm_results_are_ignored(payload):
    result = pipeline.parse_gtm(payload)
    assert result.scores is None and result.error


@pytest.mark.django_db
def test_users_cannot_open_other_users_events_but_operators_can(client):
    login(client, "alice")
    code = post(client, "siren.wav", wav_bytes(siren())).json()["code"]
    client.logout()
    login(client, "bob")
    assert client.get(reverse("events:detail", args=[code])).status_code == 403
    assert client.get(reverse("events:audio", args=[code])).status_code == 403
    client.logout()
    login(client, "sam", role=roles.SECURITY)
    assert client.get(reverse("events:detail", args=[code])).status_code == 200


@pytest.mark.django_db
def test_normal_users_cannot_batch_upload(client):
    login(client)
    response = post(client, "siren.wav", wav_bytes(siren()), batch=True)
    assert response.status_code == 403


@pytest.mark.django_db
def test_operators_batch_upload_is_marked(client):
    login(client, "rev", role=roles.REVIEWER)
    code = post(client, "siren.wav", wav_bytes(siren()), batch=True).json()["code"]
    assert SoundEvent.objects.get(code=code).record.source == AudioRecord.Source.BATCH


@pytest.mark.django_db
def test_audio_supports_range_requests_for_seeking(client):
    login(client)
    code = post(client, "siren.wav", wav_bytes(siren())).json()["code"]
    response = client.get(reverse("events:audio", args=[code]), HTTP_RANGE="bytes=100-199")
    assert response.status_code == 206 and len(response.content) == 100
    assert response["Content-Range"].startswith("bytes 100-199/")


def _test_clip():
    path = DEFAULT_METADATA_DIR / "dataset_metadata.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test" and row["class_name"] == "Glass Breaking" and (DEFAULT_OUT_DIR / row["filename"]).exists():
                return row
    return None


@pytest.mark.django_db
def test_dataset_clip_keeps_its_reference_label_for_evaluation(client):
    row = _test_clip()
    if row is None:
        pytest.skip("dataset audio not available on this machine")
    login(client, "rev", role=roles.REVIEWER)
    data = (DEFAULT_OUT_DIR / row["filename"]).read_bytes()
    code = post(client, "clip.wav", data, gtm=gtm_payload("Glass Breaking", 0.8)).json()["code"]
    record = SoundEvent.objects.get(code=code).record
    assert record.reference_class == "Glass Breaking" and record.reference_audio_id == row["audio_id"]
