"""Data export of sound-event records as CSV or Excel (FR lxx).

One row per event with its audio metadata, both models' results, the
decision, alerts and the review decision. The export uses the event-history
filters, so an administrator exports exactly the rows they searched for.

Two safety details:
- CSV is written with a UTF-8 byte-order mark, which is what makes Excel
  open it with the right characters ("Excel-compatible").
- A cell that starts with = + - @ could run as a formula when the file is
  opened in a spreadsheet (CSV/formula injection, e.g. through a file name),
  so such text gets a leading apostrophe and is shown as plain text.
"""

from __future__ import annotations

import csv
import io

from django.utils import timezone

COLUMNS = [
    ("Event ID", lambda e: e.code),
    ("Audio ID", lambda e: e.record.code),
    ("Processed at", lambda e: timezone.localtime(e.created_at).strftime("%Y-%m-%d %H:%M:%S")),
    ("Source", lambda e: e.record.get_source_display()),
    ("Live session", lambda e: e.record.session.code if e.record.session else ""),
    ("User", lambda e: e.record.owner.user_code),
    ("Username", lambda e: e.record.owner.username),
    ("File name", lambda e: e.record.original_filename),
    ("Format", lambda e: e.record.format),
    ("Duration (s)", lambda e: round(e.record.duration_s, 3)),
    ("Sampling rate (Hz)", lambda e: e.record.sample_rate),
    ("Channels", lambda e: e.record.channels),
    ("Bit depth", lambda e: e.record.bit_depth or ""),
    ("File size (bytes)", lambda e: e.record.file_size),
    ("SHA-256", lambda e: e.record.sha256),
    ("Duplicate of", lambda e: getattr(e.record.duplicate_of or e.record.near_duplicate_of, "code", "")),
    ("Audio quality", lambda e: e.record.quality_grade),
    ("Quality issues", lambda e: "; ".join(e.record.quality_issues)),
    ("Python model version", lambda e: e.python_model_version),
    ("Python prediction", lambda e: e.python_class),
    ("Python confidence", lambda e: round(e.python_confidence, 4)),
    ("Python top-two margin", lambda e: round(e.python_margin, 4)),
    ("Python top 3", lambda e: "; ".join(f"{c} {p:.3f}" for c, p in e.ranked("python", 3))),
    ("GTM model version", lambda e: e.gtm_model_version),
    ("GTM prediction", lambda e: e.gtm_class),
    ("GTM confidence", lambda e: "" if e.gtm_confidence is None else round(e.gtm_confidence, 4)),
    ("GTM top 3", lambda e: "; ".join(f"{c} {p:.3f}" for c, p in e.ranked("gtm", 3))),
    ("GTM error", lambda e: e.gtm_error),
    ("Confidence difference", lambda e: "" if e.confidence_difference is None else round(e.confidence_difference, 4)),
    ("Model agreement", lambda e: e.consistency),
    ("Final class (automatic)", lambda e: e.final_class),
    ("Confidence level", lambda e: e.confidence_level),
    ("Severity (automatic)", lambda e: e.severity),
    ("Alert status", lambda e: e.alert_status),
    ("Alerts", lambda e: "; ".join(f"{a.code} {a.get_status_display()}" for a in e.alerts.all())),
    ("Review required", lambda e: "Yes" if e.review_required else "No"),
    ("Review reasons", lambda e: "; ".join(e.review_reasons)),
    ("Status", lambda e: e.status),
    ("Reviewer decision", lambda e: e.reviewed_class),
    ("Reviewed severity", lambda e: e.reviewed_severity),
    ("Reviewed by", lambda e: e.reviewed_by.username if e.reviewed_by else ""),
    ("Reviewed at", lambda e: timezone.localtime(e.reviewed_at).strftime("%Y-%m-%d %H:%M:%S") if e.reviewed_at else ""),
    ("Recommended action", lambda e: e.effective_action),
]
HEADER = [name for name, _ in COLUMNS]
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def safe(value):
    """Text that a spreadsheet would treat as a formula is prefixed with ' so it stays text."""
    if isinstance(value, str) and value.startswith(FORMULA_START):
        return "'" + value
    return value


def rows(events):
    events = events.select_related("record__owner", "record__session", "record__duplicate_of",
                                   "record__near_duplicate_of", "reviewed_by").prefetch_related("alerts")
    for event in events.iterator(chunk_size=500):
        yield [safe(get(event)) for _, get in COLUMNS]


class _Echo:
    """A file-like object whose write() returns the line, so csv.writer can feed a streaming response."""

    def write(self, value):
        return value


def csv_lines(events):
    writer = csv.writer(_Echo())
    yield "﻿"  # byte-order mark: Excel then reads the file as UTF-8
    yield writer.writerow(HEADER)
    for row in rows(events):
        yield writer.writerow(row)


def xlsx_bytes(events) -> bytes:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font

    book = Workbook(write_only=True)
    sheet = book.create_sheet("Sound events")
    sheet.freeze_panes = "C2"
    header = [WriteOnlyCell(sheet, value=name) for name in HEADER]
    for cell in header:
        cell.font = Font(bold=True)
    sheet.append(header)
    for row in rows(events):
        sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()
