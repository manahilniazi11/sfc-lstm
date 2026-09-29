"""Downloadable analysis report (FR lxix) and data export (FR lxx)."""

import csv
import io

import pytest
from django.urls import reverse

from factories import login, make_event, make_user
from sonic.web.accounts import roles
from sonic.web.accounts.models import AuditLog
from sonic.web.events import export, report


@pytest.fixture
def events(db):
    owner = make_user("owner")
    return [make_event(owner, "Gunshot", "Critical", filename="=HYPERLINK(evil).wav"),
            make_event(owner, "Vehicle Horn", "Low", confidence=0.55, quality="Poor")]


def test_report_is_a_pdf_with_the_event(events):
    pdf = report.build(events[0], events[0].record.owner)
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000


def test_report_download_is_audited_and_limited_to_visible_events(client, events):
    url = reverse("events:report", args=[events[0].code])
    login(client, "stranger")
    assert client.get(url).status_code == 403  # another user's event
    client.login(username="owner", password="Sonic-test-2026")
    response = client.get(url)
    assert response.status_code == 200 and response["Content-Type"] == "application/pdf"
    assert f'{events[0].code}_report.pdf' in response["Content-Disposition"]
    assert AuditLog.objects.filter(action=AuditLog.Action.REPORT, details__ok=True).count() == 1


def test_report_failure_shows_a_message(client, events, monkeypatch):
    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(report, "build", broken)
    client.login(username="owner", password="Sonic-test-2026")
    response = client.get(reverse("events:report", args=[events[0].code]), follow=True)
    assert b"could not be generated" in response.content
    assert AuditLog.objects.get(action=AuditLog.Action.REPORT).details == {"ok": False, "error": "OSError"}


def test_only_administrators_export(client, events):
    for username, role in [("user1", roles.USER), ("rev", roles.REVIEWER), ("sec", roles.SECURITY)]:
        login(client, username, role)
        assert client.get(reverse("events:export", args=["csv"])).status_code == 403


def read_csv(response):
    text = b"".join(response.streaming_content).decode("utf-8")
    assert text.startswith("﻿")  # BOM so Excel reads UTF-8
    return list(csv.reader(io.StringIO(text[1:])))


def test_csv_export_uses_the_filters_and_blocks_formulas(client, events):
    login(client, "admin1", roles.ADMIN)
    response = client.get(reverse("events:export", args=["csv"]))
    rows = read_csv(response)
    assert rows[0] == export.HEADER and len(rows) == 3
    by_id = {r[0]: dict(zip(rows[0], r)) for r in rows[1:]}
    gun = by_id[events[0].code]
    assert gun["File name"] == "'=HYPERLINK(evil).wav"  # shown as text, never run as a formula
    assert gun["Python prediction"] == "Gunshot" and gun["Audio ID"] == events[0].record.code
    filtered = read_csv(client.get(reverse("events:export", args=["csv"]), {"quality": "Poor"}))
    assert [r[0] for r in filtered[1:]] == [events[1].code]
    log = AuditLog.objects.filter(action=AuditLog.Action.EXPORT).order_by("pk").last()
    assert log.details["rows"] == 1 and log.details["filters"] == {"quality": "Poor"}


def test_excel_export(client, events):
    from openpyxl import load_workbook

    login(client, "admin1", roles.ADMIN)
    response = client.get(reverse("events:export", args=["xlsx"]))
    assert response.status_code == 200 and response["Content-Disposition"].endswith('.xlsx"')
    sheet = load_workbook(io.BytesIO(response.content)).active
    values = list(sheet.values)
    assert list(values[0]) == export.HEADER and len(values) == 3


@pytest.mark.django_db
def test_unknown_export_format(client):
    login(client, "admin1", roles.ADMIN)
    assert client.get(reverse("events:export", args=["pdf"])).status_code == 404
