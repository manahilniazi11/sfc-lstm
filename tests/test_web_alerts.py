"""Alert handling, alert history, automatic escalation and the manual-review workflow (FR lv-lxii)."""

from datetime import timedelta

import pytest
from django.core.management import call_command
from django.urls import reverse
from django.utils import timezone

from sonic.detection import decision as D
from sonic.web.accounts import roles
from sonic.web.accounts.models import AuditLog, User
from sonic.web.alerts import handling
from sonic.web.alerts.models import Alert, AlertAction
from sonic.web.events import review
from sonic.web.events.models import AudioRecord, ReviewEntry, SoundEvent

PASSWORD = "Sonic-test-2026"


def make_user(username, role=roles.USER):
    return User.objects.create_user(username=username, email=f"{username}@example.com", password=PASSWORD, role=role)


def login(client, username, role=roles.USER):
    user = make_user(username, role)
    client.login(username=username, password=PASSWORD)
    return user


def make_event(owner, final_class="Gunshot", severity="Critical", review_required=False, reasons=(),
               status=D.CLASSIFIED, python_class=None, gtm_class="Gunshot"):
    record = AudioRecord.objects.create(owner=owner, original_filename="clip.wav", format="WAV", file_size=100,
                                        duration_s=2, sample_rate=16000, channels=1, sha256="0" * 64,
                                        quality_grade="Good")
    return SoundEvent.objects.create(
        record=record, python_model_version="test", python_scores={python_class or final_class: 0.8},
        python_class=python_class or final_class, python_confidence=0.8, python_margin=0.6, gtm_class=gtm_class,
        gtm_scores={gtm_class: 0.7}, gtm_confidence=0.7, consistency="Acceptable Match", final_class=final_class,
        confidence_level="High", severity=severity, alert_status="Alert generated", review_required=review_required,
        review_reasons=list(reasons), status=status, recommended_action="Alert security.",
    )


def make_alert(event, minutes=2, created_ago=0):
    alert = Alert.objects.create(event=event, category=event.final_class, severity=event.severity,
                                 alert_type="security", critical=True, message=f"{event.final_class} detected",
                                 recommended_action=event.recommended_action, escalate_after_minutes=minutes)
    AlertAction.objects.create(alert=alert, kind=AlertAction.Kind.CREATED, note=alert.message)
    if created_ago:
        Alert.objects.filter(pk=alert.pk).update(created_at=timezone.now() - timedelta(minutes=created_ago))
        alert.refresh_from_db()
    return alert


# --- alert actions ---

@pytest.mark.django_db
def test_alert_lifecycle_is_kept_in_the_history():
    operator = make_user("sec", roles.SECURITY)
    alert = make_alert(make_event(operator))
    handling.act(alert, "acknowledge", operator, "on my way")
    handling.act(alert, "resolve", operator, "checked, fireworks next door")
    alert.refresh_from_db()
    assert alert.status == Alert.Status.RESOLVED and alert.handled_by == operator
    kinds = list(alert.actions.values_list("kind", flat=True))
    assert kinds == ["created", "acknowledged", "resolved"]
    assert AuditLog.objects.filter(action=AuditLog.Action.ALERT_ACTION).count() == 2


@pytest.mark.django_db
@pytest.mark.parametrize("status, action", [
    (Alert.Status.RESOLVED, "acknowledge"), (Alert.Status.DISMISSED, "escalate"),
    (Alert.Status.ACTIVE, "resolve"), (Alert.Status.ESCALATED, "escalate"),
])
def test_impossible_transitions_are_refused(status, action):
    operator = make_user("sec", roles.SECURITY)
    alert = make_alert(make_event(operator))
    Alert.objects.filter(pk=alert.pk).update(status=status)
    alert.refresh_from_db()
    with pytest.raises(handling.ActionRefused):
        handling.act(alert, action, operator)
    alert.refresh_from_db()
    assert alert.status == status


@pytest.mark.django_db
def test_dismissing_needs_a_reason():
    operator = make_user("sec", roles.SECURITY)
    alert = make_alert(make_event(operator))
    with pytest.raises(handling.ActionRefused):
        handling.act(alert, "dismiss", operator, "  ")
    handling.act(alert, "dismiss", operator, "door slam, not a gunshot")
    alert.refresh_from_db()
    assert alert.status == Alert.Status.DISMISSED
    assert alert.actions.last().note == "door slam, not a gunshot"


@pytest.mark.django_db
def test_only_alert_handlers_can_act(client):
    owner = make_user("owner")
    alert = make_alert(make_event(owner))
    url = reverse("alerts:action", args=[alert.code, "acknowledge"])
    for username, role in [("user", roles.USER), ("rev", roles.REVIEWER)]:
        login(client, username, role)
        assert client.post(url).status_code == 403
    login(client, "maint", roles.MAINTENANCE)
    assert client.post(url).status_code == 302
    alert.refresh_from_db()
    assert alert.status == Alert.Status.ACKNOWLEDGED


@pytest.mark.django_db
def test_actions_must_be_posted(client):
    login(client, "sec", roles.SECURITY)
    alert = make_alert(make_event(make_user("owner")))
    assert client.get(reverse("alerts:action", args=[alert.code, "acknowledge"])).status_code == 405


@pytest.mark.django_db
def test_normal_users_only_see_alerts_of_their_own_recordings(client):
    alice = login(client, "alice")
    mine, theirs = make_alert(make_event(alice)), make_alert(make_event(make_user("bob")))
    page = client.get(reverse("alerts:list"))
    assert mine.code.encode() in page.content and theirs.code.encode() not in page.content
    assert client.get(reverse("alerts:detail", args=[theirs.code])).status_code == 404
    login(client, "sec", roles.SECURITY)
    page = client.get(reverse("alerts:list"))
    assert mine.code.encode() in page.content and theirs.code.encode() in page.content


@pytest.mark.django_db
def test_alert_pages_render_with_actions_for_handlers(client):
    alert = make_alert(make_event(make_user("owner")))
    login(client, "sec", roles.SECURITY)
    page = client.get(reverse("alerts:detail", args=[alert.code]))
    assert page.status_code == 200 and b"Acknowledge" in page.content and b"Dismiss" in page.content
    client.post(reverse("alerts:action", args=[alert.code, "note"]), {"note": "camera 3 checked"})
    assert b"camera 3 checked" in client.get(reverse("alerts:detail", args=[alert.code])).content
    for tab in ["attention", "handling", "closed", "all"]:
        assert client.get(reverse("alerts:list"), {"tab": tab, "severity": "Critical"}).status_code == 200


# --- automatic escalation ---

@pytest.mark.django_db
def test_unacknowledged_alerts_escalate_after_the_rule_time():
    owner = make_user("owner")
    late = make_alert(make_event(owner), minutes=2, created_ago=3)
    recent = make_alert(make_event(owner), minutes=2, created_ago=1)
    handled = make_alert(make_event(owner), minutes=2, created_ago=10)
    handling.act(handled, "acknowledge", make_user("sec", roles.SECURITY))
    assert [a.code for a in handling.escalate_overdue()] == [late.code]
    late.refresh_from_db()
    assert late.status == Alert.Status.ESCALATED
    assert late.actions.last().kind == AlertAction.Kind.AUTO_ESCALATED and late.actions.last().user is None
    assert handling.escalate_overdue() == []  # only once


@pytest.mark.django_db
def test_escalation_command(capsys):
    make_alert(make_event(make_user("owner")), minutes=1, created_ago=5)
    call_command("escalate_alerts")
    assert "1 alert(s) escalated" in capsys.readouterr().out


# --- manual review ---

@pytest.mark.django_db
def test_queue_holds_uncertain_events_most_severe_first():
    owner = make_user("owner")
    low = make_event(owner, "Vehicle Horn", "Low", True, [D.R_LOW_CONF], D.UNCERTAIN_STATUS)
    critical = make_event(owner, "Gunshot", "Critical", True, [D.R_DISAGREE], D.UNCERTAIN_STATUS)
    make_event(owner, "Gunshot", "Critical")  # no review needed
    make_event(owner, "Aggression", "High", True, [D.R_DISAGREE], D.REVIEWED)  # already reviewed
    assert list(review.queue()) == [critical, low]


@pytest.mark.django_db
def test_correction_overrides_the_result_but_keeps_the_model_outputs():
    reviewer = make_user("rev", roles.REVIEWER)
    event = make_event(make_user("owner"), "Gunshot", "Critical", True, [D.R_DISAGREE], D.UNCERTAIN_STATUS,
                       gtm_class="Glass Breaking")
    alert = make_alert(event)
    review.decide(event, reviewer, review.ReviewDecision("correct", "Glass Breaking", comment="clearly glass"))
    event.refresh_from_db()
    assert event.effective_class == "Glass Breaking" and event.status == D.REVIEWED
    assert event.effective_severity == "High"  # the corrected class's rule
    # the original outputs are untouched
    assert (event.final_class, event.python_class, event.gtm_class, event.severity) == \
        ("Gunshot", "Gunshot", "Glass Breaking", "Critical")
    entry = event.reviews.get()
    assert entry.kind == ReviewEntry.Kind.CORRECT and entry.previous_class == "Gunshot" and entry.comment == "clearly glass"
    log = AuditLog.objects.get(action=AuditLog.Action.OVERRIDE)
    assert log.details["automatic"] == "Gunshot" and log.details["reviewed"] == "Glass Breaking"
    assert "corrected the sound to Glass Breaking" in alert.actions.last().note
    assert event not in review.queue()


@pytest.mark.django_db
def test_confirmation_with_overrides_and_close():
    reviewer = make_user("rev", roles.REVIEWER)
    event = make_event(make_user("owner"), "Aggression", "High", True, [D.R_LOW_CONF], D.UNCERTAIN_STATUS)
    review.decide(event, reviewer, review.ReviewDecision("confirm", severity="Critical",
                                                         recommended_action="Call the police.", close=True))
    event.refresh_from_db()
    assert event.reviewed_class == "Aggression" and event.status == D.CLOSED
    assert event.effective_severity == "Critical" and event.effective_action == "Call the police."
    assert event.severity == "High" and event.recommended_action == "Alert security."
    assert list(event.reviews.values_list("kind", flat=True)) == ["confirm", "close"]
    review.reopen(event, reviewer)
    event.refresh_from_db()
    assert event.status == D.MANUAL_REVIEW and event in review.queue()


@pytest.mark.django_db
def test_unknown_sounds_cannot_be_confirmed_only_corrected():
    reviewer = make_user("rev", roles.REVIEWER)
    event = make_event(make_user("owner"), "Unknown", "Informational", True, [D.R_UNSUPPORTED], D.UNCERTAIN_STATUS)
    with pytest.raises(review.ReviewRefused):
        review.decide(event, reviewer, review.ReviewDecision("confirm"))
    with pytest.raises(review.ReviewRefused):
        review.decide(event, reviewer, review.ReviewDecision("correct", "Not a class"))
    review.decide(event, reviewer, review.ReviewDecision("correct", "Person Asking for Help"))
    event.refresh_from_db()
    assert event.effective_class == "Person Asking for Help" and event.effective_severity == "Critical"


@pytest.mark.django_db
def test_comments_and_recommended_actions():
    reviewer = make_user("rev", roles.REVIEWER)
    event = make_event(make_user("owner"))
    with pytest.raises(review.ReviewRefused):
        review.comment(event, reviewer, "  ")
    review.comment(event, reviewer, "sounds distant", "Check the car park camera.")
    event.refresh_from_db()
    assert event.effective_action == "Check the car park camera." and event.reviews.get().comment == "sounds distant"


@pytest.mark.django_db
def test_review_pages_need_the_review_role(client):
    event = make_event(make_user("owner"), review_required=True, reasons=[D.R_DISAGREE], status=D.UNCERTAIN_STATUS)
    login(client, "sec", roles.SECURITY)
    assert client.get(reverse("events:review_queue")).status_code == 403
    assert client.post(reverse("events:review", args=[event.code]), {"decision": "confirm"}).status_code == 403
    login(client, "rev", roles.REVIEWER)
    page = client.get(reverse("events:review_queue"))
    assert page.status_code == 200 and event.code.encode() in page.content
    assert b"Save decision" in client.get(reverse("events:detail", args=[event.code])).content


@pytest.mark.django_db
def test_save_and_next_walks_through_the_queue(client):
    owner = make_user("owner")
    first = make_event(owner, "Gunshot", "Critical", True, [D.R_DISAGREE], D.UNCERTAIN_STATUS)
    second = make_event(owner, "Vehicle Horn", "Low", True, [D.R_LOW_CONF], D.UNCERTAIN_STATUS)
    login(client, "rev", roles.REVIEWER)
    response = client.post(reverse("events:review", args=[first.code]), {"decision": "confirm", "then": "next"})
    assert response.url == reverse("events:detail", args=[second.code]) + "?from=queue"
    response = client.post(reverse("events:review", args=[second.code]),
                           {"decision": "correct", "category": "Alarm or Siren", "then": "next"})
    assert response.url == reverse("events:review_queue")
    second.refresh_from_db()
    assert second.reviewed_class == "Alarm or Siren" and second.reviewed_by.username == "rev"


@pytest.mark.django_db
def test_a_correction_without_a_class_is_refused(client):
    event = make_event(make_user("owner"), review_required=True, reasons=[D.R_DISAGREE], status=D.UNCERTAIN_STATUS)
    login(client, "rev", roles.REVIEWER)
    client.post(reverse("events:review", args=[event.code]), {"decision": "correct"})
    event.refresh_from_db()
    assert event.reviewed_class == "" and event.status == D.UNCERTAIN_STATUS


@pytest.mark.django_db
def test_menu_badges_count_open_alerts_and_waiting_reviews(client):
    owner = make_user("owner")
    make_alert(make_event(owner))
    make_event(owner, review_required=True, reasons=[D.R_DISAGREE], status=D.UNCERTAIN_STATUS)
    login(client, "admin1", roles.ADMIN)
    response = client.get(reverse("events:history"))
    assert response.context["open_alerts"] == 1 and response.context["review_waiting"] == 1
