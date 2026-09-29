"""Dashboards and analytics (FR lxiii, lxv, lxviii): the numbers, and who sees what."""

from datetime import timedelta

import pytest
from django.urls import reverse
from django.utils import timezone

from factories import login, make_alert, make_event, make_user
from sonic.detection import decision as D
from sonic.web.accounts import roles
from sonic.web.alerts import handling
from sonic.web.alerts.models import Alert, AlertAction
from sonic.web.events import stats
from sonic.web.events.models import SoundEvent


@pytest.fixture
def data(db):
    owner = make_user("owner")
    gun = make_event(owner, "Gunshot", "Critical", confidence=0.9)
    make_event(owner, "Gunshot", "Critical", confidence=0.95, gtm_class="Glass Breaking")
    make_event(owner, "Vehicle Horn", "Low", confidence=0.45, quality="Poor", days_ago=2)
    make_event(owner, "Glass Breaking", "High", confidence=0.6, reviewed_class="Vehicle Horn", reviewed_severity="Low")
    make_event(owner, "Animal Sound", "Low", confidence=0.7, gtm_class="")  # GTM gave no result
    return {"owner": owner, "gun": gun}


def test_overview(data):
    totals = stats.overview(SoundEvent.objects.all(), Alert.objects.all())
    assert totals["total"] == 5 and totals["critical"] == 2 and totals["poor_quality"] == 1
    assert totals["compared"] == 4 and totals["disagreements"] == 1 and totals["disagreement_rate"] == 0.25
    assert totals["avg_confidence"] == pytest.approx((0.9 + 0.95 + 0.45 + 0.6 + 0.7) / 5)


def test_categories_follow_review_decisions(data):
    assert dict(stats.by_category(SoundEvent.objects.all())) == {"Gunshot": 2, "Vehicle Horn": 2, "Animal Sound": 1}


def test_daily_trend_is_zero_filled(data):
    trend = stats.daily(SoundEvent.objects.all(), days=7)
    assert len(trend["labels"]) == 7 and sum(trend["events"]) == 5
    assert trend["events"][-1] == 4 and trend["events"][-3] == 1 and trend["critical"][-1] == 2


def test_confidence_histogram(data):
    hist = stats.confidence_histogram(SoundEvent.objects.all())
    assert sum(hist["python"]) == 5 and hist["python"][9] == 2 and hist["python"][4] == 1  # 90% and 95% share the top band
    assert sum(hist["gtm"]) == 4  # the event without a GTM result is left out


def test_false_positives_and_negatives(data):
    make_alert(data["gun"], status=Alert.Status.DISMISSED)
    labelled = make_event(data["owner"], "Gunshot", "Critical", gtm_class="Glass Breaking")
    labelled.record.reference_class = "Glass Breaking"
    labelled.record.save()
    errors = stats.errors_by_class(SoundEvent.objects.all(), Alert.objects.all())
    rows = {r["category"]: r for r in errors["rows"]}
    assert rows["Glass Breaking"]["fp"] == 1 and rows["Vehicle Horn"]["fn"] == 1  # the reviewer's correction
    assert rows["Gunshot"]["dismissed"] == 1
    assert rows["Gunshot"]["py_fp"] == 1 and rows["Glass Breaking"]["py_fn"] == 1
    assert errors["evaluation"]["python_accuracy"] == 0 and errors["evaluation"]["gtm_accuracy"] == 1


def test_alert_response_times(data):
    operator = make_user("sec", roles.SECURITY)
    quick = make_alert(data["gun"], created_ago=10)
    handling.act(quick, "acknowledge", operator)
    AlertAction.objects.filter(alert=quick, kind="acknowledged").update(created_at=quick.created_at + timedelta(minutes=4))
    make_alert(data["gun"], created_ago=30, status=Alert.Status.ESCALATED)
    [critical] = stats.alert_response(Alert.objects.all())
    assert critical["alerts"] == 2 and critical["handled"] == 1 and critical["median_minutes"] == pytest.approx(4)
    assert critical["open"] == 1


def test_dashboard_scope(client, data):
    login(client, "somebody")
    page = client.get(reverse("home"))
    assert page.status_code == 200 and page.context["totals"]["total"] == 0  # not their recordings
    assert "trend" not in page.context  # the system charts are for analytics roles
    client.login(username="owner", password="Sonic-test-2026")
    assert client.get(reverse("home")).context["totals"]["total"] == 5
    login(client, "admin1", roles.ADMIN)
    page = client.get(reverse("home"))
    assert page.context["totals"]["total"] == 5 and sum(page.context["trend"]["events"]) == 5


def test_analytics_access_and_filters(client, data):
    login(client, "sec", roles.SECURITY)
    assert client.get(reverse("analytics")).status_code == 403
    login(client, "rev", roles.REVIEWER)
    page = client.get(reverse("analytics"))
    assert page.status_code == 200 and page.context["totals"]["total"] == 5
    today = timezone.localdate()
    page = client.get(reverse("analytics"), {"date_from": str(today)})
    assert page.context["totals"]["total"] == 4


@pytest.mark.django_db
def test_empty_system_renders(client):
    login(client, "admin1", roles.ADMIN)
    assert client.get(reverse("home")).status_code == 200
    assert client.get(reverse("analytics")).status_code == 200


def test_waiting_list_uses_the_queue_rule(data):
    make_event(data["owner"], review_required=True, reasons=[D.R_DISAGREE], status=D.REVIEWED)
    assert stats.overview(SoundEvent.objects.all(), Alert.objects.all())["waiting"] == 0
