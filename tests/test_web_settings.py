"""Administrator settings: thresholds, alert rules and data retention (FR xxxv, xxxvi, liii, lxxx)."""

import json
from datetime import timedelta

import pytest
from django.core.management import call_command
from django.urls import reverse
from django.utils import timezone

from factories import login, make_alert, make_event, make_user
from sonic.detection import decision as D
from sonic.web.accounts import roles
from sonic.web.accounts.models import AuditLog
from sonic.web.alerts import retention
from sonic.web.alerts.models import Alert, AlertRule, DecisionSettings, current_ruleset
from sonic.web.events.models import AudioRecord, SoundEvent


def settings_data(**changes):
    config = DecisionSettings.load()
    data = {f: getattr(config, f) for f in ["min_confidence", "top_two_margin", "unknown_confidence",
                                            "weak_match_difference", "overlap_threshold", "repeat_window_seconds",
                                            "live_window_seconds", "max_upload_mb", "audio_retention_days",
                                            "event_retention_days"]}
    data["store_live_audio"] = "on"
    data.update(changes)
    return data


@pytest.mark.django_db
def test_only_administrators_open_settings(client):
    for username, role in [("u", roles.USER), ("rev", roles.REVIEWER), ("sec", roles.SECURITY)]:
        login(client, username, role)
        assert client.get(reverse("settings:decision")).status_code == 403
        assert client.get(reverse("settings:rules")).status_code == 403


@pytest.mark.django_db
def test_threshold_change_is_used_and_audited(client):
    login(client, "admin1", roles.ADMIN)
    response = client.post(reverse("settings:decision"), settings_data(min_confidence=0.65))
    assert response.status_code == 302
    assert current_ruleset().thresholds.min_confidence == 0.65
    log = AuditLog.objects.get(action=AuditLog.Action.SETTINGS)
    assert log.details["changed"]["min_confidence"][1] == 0.65


@pytest.mark.django_db
@pytest.mark.parametrize("field, value", [
    ("min_confidence", 1.5), ("unknown_confidence", 0.9), ("live_window_seconds", 5),
    ("max_upload_mb", 0), ("audio_retention_days", 400),  # longer than the event records (365)
])
def test_invalid_settings_are_refused(client, field, value):
    login(client, "admin1", roles.ADMIN)
    before = DecisionSettings.load().__dict__.copy()
    response = client.post(reverse("settings:decision"), settings_data(**{field: value}))
    assert response.status_code == 200 and response.context["form"].errors.get(field)
    assert getattr(DecisionSettings.load(), field) == before[field]


def rule_data(rule, **changes):
    data = {f: getattr(rule, f) for f in ["severity", "alert_type", "min_confidence", "top_two_margin",
                                          "consecutive_detections", "min_quality", "recommended_action", "manual_review"]}
    for flag in ["enabled", "critical", "alert", "require_model_agreement"]:
        if getattr(rule, flag):
            data[flag] = "on"
    data["escalate_after_minutes"] = rule.escalate_after_minutes or ""
    data["raise_to_critical"] = json.dumps(rule.raise_to_critical) if rule.raise_to_critical else ""
    data["noise_limit_dbfs"] = rule.noise_limit_dbfs if rule.noise_limit_dbfs is not None else ""
    data["above_noise_limit"] = json.dumps(rule.above_noise_limit) if rule.above_noise_limit else ""
    data.update(changes)
    return data


@pytest.mark.django_db
def test_rule_edit_changes_the_decision(client):
    login(client, "admin1", roles.ADMIN)
    assert client.get(reverse("settings:rules")).status_code == 200
    rule = AlertRule.objects.get(category="Gunshot")
    url = reverse("settings:rule", args=["Gunshot"])
    response = client.post(url, rule_data(rule, consecutive_detections=3, require_model_agreement="on"))
    assert response.status_code == 302
    gun = current_ruleset().for_category("Gunshot")
    assert gun.consecutive_detections == 3 and gun.require_model_agreement
    assert AuditLog.objects.get(action=AuditLog.Action.SETTINGS).details["rule"] == "Gunshot"


@pytest.mark.django_db
@pytest.mark.parametrize("changes", [
    {"consecutive_detections": 0}, {"min_confidence": 2}, {"raise_to_critical": "[1, 2]"},
    {"escalate_after_minutes": -5},
])
def test_invalid_rules_are_refused(client, changes):
    login(client, "admin1", roles.ADMIN)
    current_ruleset()
    rule = AlertRule.objects.get(category="Aggression")
    response = client.post(reverse("settings:rule", args=["Aggression"]), rule_data(rule, **changes))
    assert response.status_code == 200 and response.context["form"].errors
    rule.refresh_from_db()
    assert rule.consecutive_detections == 1 and rule.min_confidence == 0.45


@pytest.mark.django_db
def test_rules_export_round_trips(client):
    login(client, "admin1", roles.ADMIN)
    response = client.get(reverse("settings:rules_export"))
    data = json.loads(response.content)
    assert {r["category"] for r in data["rules"]} >= {"Gunshot", "Unknown"} and "min_confidence" in data["settings"]
    assert AuditLog.objects.filter(action=AuditLog.Action.EXPORT).exists()


# --- retention ---

@pytest.fixture
def old_records(db, settings, tmp_path):
    settings.UPLOAD_DIR = tmp_path
    owner = make_user("owner")

    def stored(event, days):
        folder = tmp_path / event.record.code
        folder.mkdir()
        (folder / "original.wav").write_bytes(b"RIFF")
        AudioRecord.objects.filter(pk=event.record.pk).update(
            stored_file=f"{event.record.code}/original.wav", uploaded_at=timezone.now() - timedelta(days=days))
        event.record.refresh_from_db()
        return event

    return {
        "recent": stored(make_event(owner), 5),
        "old_audio": stored(make_event(owner), 120),  # past the 90-day audio limit
        "ancient": stored(make_event(owner), 400),  # past the 365-day event limit
        "ancient_waiting": stored(make_event(owner, review_required=True, reasons=[D.R_DISAGREE],
                                             status=D.UNCERTAIN_STATUS), 400),
        "ancient_alert": stored(make_event(owner), 400),
        "tmp": tmp_path,
    }


def test_retention_plan_and_apply(old_records):
    make_alert(old_records["ancient_alert"], status=Alert.Status.ACTIVE)
    plan = retention.plan()
    assert (plan.audio_files, plan.event_records, plan.kept_open) == (1, 1, 2)
    retention.apply()
    assert not SoundEvent.objects.filter(pk=old_records["ancient"].pk).exists()
    assert not (old_records["tmp"] / old_records["ancient"].record.code).exists()
    old = AudioRecord.objects.get(pk=old_records["old_audio"].record.pk)
    assert old.stored_file == "" and not (old_records["tmp"] / old.code).exists() and old.event  # event kept
    assert AudioRecord.objects.get(pk=old_records["recent"].record.pk).stored_file  # recent audio untouched
    for kept in ["ancient_waiting", "ancient_alert"]:
        assert (old_records["tmp"] / old_records[kept].record.code / "original.wav").exists()
    log = AuditLog.objects.get(action=AuditLog.Action.RETENTION)
    assert log.details == {"audio_files": 1, "event_records": 1, "kept_open": 2}


def test_retention_never_deletes_the_upload_folder(old_records):
    event = old_records["ancient"]
    AudioRecord.objects.filter(pk=event.record.pk).update(stored_file="original.wav")  # a malformed path
    retention.apply()
    assert old_records["tmp"].exists() and (old_records["tmp"] / old_records["recent"].record.code).exists()


def test_retention_command_dry_run(old_records, capsys):
    call_command("apply_retention", "--dry-run")
    assert "would delete 1 audio file(s) and 2 event record(s); kept 1" in capsys.readouterr().out
    assert SoundEvent.objects.count() == 5


def test_retention_button(client, old_records):
    login(client, "admin1", roles.ADMIN)
    assert client.get(reverse("settings:retention")).status_code == 405
    response = client.post(reverse("settings:retention"), follow=True)
    assert b"Retention applied" in response.content and SoundEvent.objects.count() == 3
