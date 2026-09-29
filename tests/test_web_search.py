"""Search, filtering and the event timeline (FR lxvi, lxvii)."""

import pytest
from django.urls import reverse

from factories import login, make_event, make_user
from sonic.detection import decision as D
from sonic.web.accounts import roles
from sonic.web.events import search
from sonic.web.events.models import SoundEvent


def found(viewer, **params):
    form = search.EventFilterForm(params, user=viewer)
    assert form.is_valid(), form.errors
    return set(search.apply(SoundEvent.objects.all(), form).values_list("code", flat=True))


@pytest.fixture
def events(db):
    alice, bob = make_user("alice"), make_user("bob")
    return {
        "gun": make_event(alice, "Gunshot", "Critical", confidence=0.95, filename="car_park.wav"),
        "horn": make_event(alice, "Vehicle Horn", "Low", confidence=0.55, quality="Poor", days_ago=3),
        "glass": make_event(bob, "Glass Breaking", "High", confidence=0.40, filename="window.mp3", days_ago=10,
                            review_required=True, reasons=[D.R_LOW_CONF], status=D.UNCERTAIN_STATUS),
        "fixed": make_event(bob, "Gunshot", "Critical", confidence=0.7, reviewed_class="Glass Breaking",
                            reviewed_severity="High", status=D.REVIEWED),
    }


def codes(events, *names):
    return {events[n].code for n in names}


def test_each_filter(events):
    admin = make_user("admin1", roles.ADMIN)
    assert found(admin, q=events["gun"].record.code) == codes(events, "gun")  # Audio ID
    assert found(admin, q=events["horn"].code) == codes(events, "horn")  # event ID
    assert found(admin, q="WINDOW") == codes(events, "glass")  # file name, any case
    assert found(admin, severity="Low") == codes(events, "horn")
    assert found(admin, quality="Poor") == codes(events, "horn")
    assert found(admin, user=events["glass"].record.owner.user_code) == codes(events, "glass", "fixed")
    assert found(admin, conf_min=60) == codes(events, "gun", "fixed")
    assert found(admin, conf_min=50, conf_max=60) == codes(events, "horn")


def test_category_uses_the_reviewers_decision(events):
    admin = make_user("admin1", roles.ADMIN)
    assert found(admin, category="Glass Breaking") == codes(events, "glass", "fixed")
    assert found(admin, category="Gunshot") == codes(events, "gun")


def test_review_status(events):
    admin = make_user("admin1", roles.ADMIN)
    assert found(admin, review="waiting") == codes(events, "glass")
    assert found(admin, review="corrected") == codes(events, "fixed")
    assert found(admin, review="not_needed") == codes(events, "gun", "horn", "fixed")


def test_date_range(events):
    from django.utils import timezone

    admin = make_user("admin1", roles.ADMIN)
    today = timezone.localdate()
    assert found(admin, date_from=str(today)) == codes(events, "gun", "fixed")
    week_ago = today - timezone.timedelta(days=7)
    assert found(admin, date_to=str(week_ago)) == codes(events, "glass")


@pytest.mark.django_db
def test_invalid_ranges_are_reported():
    form = search.EventFilterForm({"conf_min": 80, "conf_max": 20, "date_from": "2026-09-10", "date_to": "2026-09-01"},
                                  user=make_user("admin1", roles.ADMIN))
    assert not form.is_valid() and "conf_max" in form.errors and "date_to" in form.errors


def test_normal_users_search_only_their_own_events_and_have_no_user_filter(client, events):
    client.login(username="alice", password="Sonic-test-2026")
    page = client.get(reverse("events:history"), {"category": "Glass Breaking"})
    assert "user" not in page.context["form"].fields
    assert page.context["total"] == 0  # bob's glass events are not alice's
    page = client.get(reverse("events:history"))
    assert page.context["total"] == 2


def test_history_table_and_timeline_render(client, events):
    login(client, "rev", roles.REVIEWER)
    table = client.get(reverse("events:history"), {"severity": "Critical"})
    assert table.status_code == 200 and table.context["total"] == 1  # the corrected one is High now
    timeline = client.get(reverse("events:history"), {"view": "timeline"})
    assert timeline.status_code == 200 and b"timeline-item" in timeline.content
    days = [day for day, _ in timeline.context["days"]]
    assert days == sorted(days, reverse=True) and len(days) == 3
