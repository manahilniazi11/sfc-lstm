"""The downloadable analysis report of one sound event, as a PDF (FR lxix).

It contains everything the SRS lists: audio metadata, both models'
predictions and confidence scores, the confidence difference, the waveform
and spectrogram, audio quality, severity, alert status and the review
decision, plus the model versions and the decision's reasons so the printed
report can be understood without the app.

Built with ReportLab's "platypus" layout: a list of paragraphs, tables and
images that ReportLab flows onto A4 pages.
"""

from __future__ import annotations

import io
from pathlib import Path

from django.conf import settings
from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from . import explain
from .models import SoundEvent

GRID = colors.HexColor("#d0d5dd")
HEAD = colors.HexColor("#eef2f7")
SEVERITY_COLOURS = {"Informational": "#6c757d", "Low": "#198754", "Medium": "#b58500", "High": "#d9480f",
                    "Critical": "#dc3545"}

styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#52514e"))


def p(text, style=BODY) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def pct(value) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def table(rows, widths, header=False) -> Table:
    t = Table([[c if isinstance(c, Paragraph) else p(c) for c in row] for row in rows], colWidths=widths)
    style = [("GRID", (0, 0), (-1, -1), 0.5, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    else:
        style.append(("BACKGROUND", (0, 0), (0, -1), HEAD))
    t.setStyle(TableStyle(style))
    return t


def facts(pairs) -> Table:
    return table([[label, value] for label, value in pairs], [55 * mm, 115 * mm])


def image(path: Path, width_mm: float = 170) -> Image | None:
    if not path.is_file():
        return None
    picture = Image(str(path))
    scale = width_mm * mm / picture.imageWidth
    picture.drawWidth, picture.drawHeight = picture.imageWidth * scale, picture.imageHeight * scale
    return picture


def build(event: SoundEvent, generated_by) -> bytes:
    record = event.record
    story = []
    severity_colour = SEVERITY_COLOURS.get(event.effective_severity, "#000000")
    story += [
        p("SonicSentinel AI - sound event analysis report", SMALL),
        Paragraph(f"{escape(event.effective_class)} "
                  f"<font color='{severity_colour}'>[{escape(event.effective_severity)}]</font>", H1),
        p(f"{event.code} | Audio ID {record.code} | {record.get_source_display()} | "
          f"analysed {timezone.localtime(event.created_at):%Y-%m-%d %H:%M:%S} | "
          f"report generated {timezone.localtime():%Y-%m-%d %H:%M} by {generated_by.username}", SMALL),
        Spacer(1, 4 * mm),
    ]

    story.append(p("Summary", H2))
    for line in explain.summary(event):
        story.append(p(f"- {line}"))
    story.append(p(f"Recommended action: {event.effective_action or '-'}"))

    story.append(p("Final decision", H2))
    story.append(facts([
        ("Detected sound category", event.final_class),
        ("Reviewer decision", f"{event.reviewed_class} (by {event.reviewed_by.username if event.reviewed_by else '-'}, "
                              f"{timezone.localtime(event.reviewed_at):%Y-%m-%d %H:%M})" if event.reviewed_class
                              else "Not reviewed"),
        ("Model agreement", f"{event.consistency} - {explain.CONSISTENCY_MEANING.get(event.consistency, '')}"),
        ("Confidence level", f"{event.confidence_level} ({pct(event.python_confidence)})"),
        ("Severity", event.effective_severity + (f" (automatic: {event.severity})" if event.reviewed_severity else "")),
        ("Alert status", event.alert_status + "".join(f"; {a.code} {a.get_status_display()}" for a in event.alerts.all())),
        ("Manual review", "; ".join(event.review_reasons) or "Not required"),
        ("Event status", event.status),
        ("Repeated detections", str(event.repeated_detections)),
        ("Overlapping sounds", ", ".join(event.overlapping_classes) or "None"),
    ]))

    story.append(p("Model predictions and confidence scores", H2))
    story.append(table([
        ["", "Python model", "Google Teachable Machine"],
        ["Predicted class", event.python_class, event.gtm_class or "no result"],
        ["Top-class confidence", pct(event.python_confidence), pct(event.gtm_confidence)],
        ["Top-two margin", pct(event.python_margin), pct(event.gtm_margin)],
        ["Model version", event.python_model_version, event.gtm_model_version or "-"],
    ], [45 * mm, 62 * mm, 63 * mm], header=True))
    story.append(Spacer(1, 2 * mm))
    difference = pct(event.confidence_difference) if event.confidence_difference is not None else "-"
    story.append(p(f"Confidence difference |Python - GTM| = {difference}. "
                   f"Predicted classes {'match' if event.class_match else 'differ'}."))
    if event.gtm_error:
        story.append(p(f"GTM: {event.gtm_error}", SMALL))
    rows = [["Class", "Python", "GTM", "Difference"]]
    for row in explain.comparison_rows(event):
        rows.append([row["name"], pct(row["python"]), pct(row["gtm"]), pct(row["difference"])])
    story.append(Spacer(1, 2 * mm))
    story.append(table(rows, [70 * mm, 33 * mm, 33 * mm, 34 * mm], header=True))

    story.append(p("Audio metadata", H2))
    story.append(facts([
        ("File name", record.original_filename),
        ("Format", record.format),
        ("Duration", f"{record.duration_s:.2f} s"),
        ("Sampling rate", f"{record.sample_rate} Hz"),
        ("Channels", str(record.channels)),
        ("Bit depth", f"{record.bit_depth}-bit" if record.bit_depth else "n/a (compressed format)"),
        ("File size", f"{record.file_size:,} bytes"),
        ("Uploaded", f"{timezone.localtime(record.uploaded_at):%Y-%m-%d %H:%M:%S} by {record.owner.username} ({record.owner.user_code})"),
        ("SHA-256", record.sha256),
        ("Duplicate", f"identical to {record.duplicate_of.code}" if record.duplicate_of
         else f"near-duplicate of {record.near_duplicate_of.code}" if record.near_duplicate_of else "No"),
    ]))

    story.append(p("Audio quality", H2))
    m = record.quality_metrics or {}
    story.append(facts([
        ("Grade", record.quality_grade),
        ("Problems found", "; ".join(record.quality_issues) or "None"),
        ("Peak level", f"{m['peak_dbfs']} dBFS" if m.get("peak_dbfs") is not None else "-"),
        ("Background noise estimate", f"{m['noise_floor_dbfs']} dBFS" if m.get("noise_floor_dbfs") is not None else "-"),
        ("Clipped samples", pct(m.get("clipped_fraction"))),
        ("Silent part", pct(m.get("silent_fraction"))),
    ]))

    story.append(p("Waveform and spectrogram", H2))
    if record.stored_file:
        folder = (Path(settings.UPLOAD_DIR) / record.stored_file).parent
        for name, caption in [("waveform.png", "Waveform (shaded: windows where the Python model heard an event)"),
                              ("spectrogram.png", "Mel spectrogram, 0-8 kHz (brighter = louder at that pitch)")]:
            picture = image(folder / name)
            if picture:
                story += [p(caption, SMALL), picture, Spacer(1, 3 * mm)]
    else:
        story.append(p("The audio of this record was not kept, so no images are available.", SMALL))

    reviews = list(event.reviews.select_related("reviewer"))
    if reviews:
        story.append(p("Review history", H2))
        rows = [["When", "Reviewer", "Action", "Details"]]
        for r in reviews:
            details = "; ".join(x for x in [
                f"{r.previous_class} -> {r.new_class}" if r.kind == "correct" else r.new_class,
                f"severity {r.severity}" if r.severity else "", r.recommended_action, r.comment] if x)
            rows.append([f"{timezone.localtime(r.created_at):%Y-%m-%d %H:%M}", r.reviewer.username if r.reviewer else "-",
                         r.get_kind_display(), details])
        story.append(table(rows, [30 * mm, 28 * mm, 35 * mm, 77 * mm], header=True))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=15 * mm,
                            bottomMargin=15 * mm, title=f"{event.code} analysis report", author="SonicSentinel AI")
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return buffer.getvalue()


def _page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6c757d"))
    canvas.drawString(20 * mm, 8 * mm, "Competition prototype for controlled, ethical testing; "
                                       "not a certified emergency-response system.")
    canvas.drawRightString(A4[0] - 20 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()
