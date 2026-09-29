from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.http import FileResponse, Http404, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.templatetags.static import static
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from sonic.detection import decision as D
from sonic.detection.rules import SEVERITIES

from ..accounts.audit import Action, record as audit
from ..accounts.decorators import capability_required
from ..alerts.models import DecisionSettings
from . import explain, export, report, review, search
from .files import ranged_file_response
from .forms import CommentForm, ReviewForm
from .gtm import gtm_info
from .models import AudioRecord, SoundEvent
from .pipeline import UploadRejected, analyse_file, parse_gtm, save_upload


def visible_events(user):
    """Operators see every event; a normal user sees their own (FR ii)."""
    events = SoundEvent.objects.select_related("record", "record__owner")
    return events if user.can("view_all_events") else events.filter(record__owner=user)


def get_event_for(user, code: str) -> SoundEvent:
    event = get_object_or_404(SoundEvent.objects.select_related("record", "record__owner"), code=code)
    if event.record.owner_id != user.pk and not user.can("view_all_events"):
        raise PermissionDenied("This event belongs to another user.")
    return event


def gtm_client_config() -> dict:
    """Where the browser loads the GTM model from, and the version it reports back."""
    info = gtm_info()
    return {"gtm_available": info["available"], "gtm_url": static("gtm_model/"), "gtm_version": info["version"]}


@capability_required("analyse")
def upload(request):
    limits = DecisionSettings.load()
    batch = request.user.can("batch_upload")
    config = {**gtm_client_config(), "analyse_url": reverse("events:analyse"), "max_mb": limits.max_upload_mb, "batch": batch}
    return render(request, "events/upload.html", {
        "gtm": gtm_info(), "max_mb": limits.max_upload_mb, "batch": batch, "upload_config": config,
    })


@require_POST
@capability_required("analyse")
def analyse(request):
    """Receives one file plus the GTM result the browser computed for it.

    A batch upload is the page sending its files one request at a time,
    because each file carries its own GTM result.
    """
    wants_json = request.headers.get("x-requested-with") == "fetch"
    files = request.FILES.getlist("audio")
    if len(files) != 1:
        return _upload_result(request, wants_json, {"ok": False, "file": "", "error": "Send exactly one audio file per request."})
    uploaded = files[0]
    batch = request.POST.get("batch") == "1"
    if batch and not request.user.can("batch_upload"):
        return _upload_result(request, wants_json, {"ok": False, "file": uploaded.name,
                                                    "error": "Your role cannot use batch upload."}, status=403)
    source = AudioRecord.Source.BATCH if batch else AudioRecord.Source.UPLOAD
    try:
        path, sha256, size = save_upload(uploaded, DecisionSettings.load().max_upload_mb)
        event = analyse_file(path, original_name=uploaded.name, sha256=sha256, size=size, user=request.user,
                             gtm=parse_gtm(request.POST.get("gtm")), source=source, request=request)
    except UploadRejected as exc:
        return _upload_result(request, wants_json, {"ok": False, "file": uploaded.name, "error": str(exc)}, status=400)
    return _upload_result(request, wants_json, {
        "ok": True, "file": uploaded.name, "code": event.code, "url": reverse("events:detail", args=[event.code]),
        "final_class": event.final_class, "severity": event.severity, "status": event.status,
        "alert": event.alert_status, "duplicate": event.record.is_duplicate,
    })


def _upload_result(request, wants_json, result: dict, status: int = 200):
    if wants_json:
        return JsonResponse(result, status=status)
    if result["ok"]:
        return redirect(result["url"])
    messages.error(request, f"{result['file']}: {result['error']}" if result["file"] else result["error"])
    return redirect("events:upload")


@login_required
def detail(request, code: str):
    event = get_event_for(request.user, code)
    return render(request, "events/detail.html", {
        "event": event,
        "record": event.record,
        "summary": explain.summary(event),
        "confidence_note": explain.confidence_note(event),
        "consistency_meaning": explain.CONSISTENCY_MEANING.get(event.consistency, ""),
        "comparison_rows": explain.comparison_rows(event),
        "python_top": event.ranked("python", 3),
        "gtm_top": event.ranked("gtm", 3),
        "prep": event.windows.get("preprocessing", {}),
        "alerts": event.alerts.all(),
        "reviews": event.reviews.select_related("reviewer"),
        "review_form": ReviewForm(event=event) if request.user.can("review") else None,
        "comment_form": CommentForm() if request.user.can("review") else None,
        "in_queue": request.GET.get("from") == "queue",
    })


def _stored_path(record: AudioRecord, name: str) -> Path:
    if not record.stored_file:
        raise Http404("the audio of this record was not kept")
    folder = (Path(settings.UPLOAD_DIR) / record.stored_file).parent
    return folder / name if name else Path(settings.UPLOAD_DIR) / record.stored_file


@login_required
def audio(request, code: str):
    event = get_event_for(request.user, code)
    return ranged_file_response(request, _stored_path(event.record, ""))


@login_required
def image(request, code: str, kind: str):
    if kind not in ("waveform", "spectrogram"):
        raise Http404()
    event = get_event_for(request.user, code)
    path = _stored_path(event.record, f"{kind}.png")
    if not path.is_file():
        raise Http404("image not available")
    response = FileResponse(open(path, "rb"), content_type="image/png")
    response["Cache-Control"] = "private, max-age=3600"
    return response


@login_required
def history(request):
    form = search.EventFilterForm(request.GET or None, user=request.user)
    events = search.apply(visible_events(request.user).select_related("record__session"), form)
    view = "timeline" if request.GET.get("view") == "timeline" else "table"
    page = Paginator(events, 50 if view == "timeline" else 25).get_page(request.GET.get("page"))
    days = []  # the timeline groups the page's events by day
    for event in page:
        day = timezone.localdate(event.created_at)
        if not days or days[-1][0] != day:
            days.append((day, []))
        days[-1][1].append(event)
    return render(request, "events/history.html", {
        "form": form, "page": page, "view": view, "days": days, "total": page.paginator.count,
        "filters_active": len(form.active) if request.GET else 0,
        "query": request.GET.urlencode(),
    })


@login_required
def event_report(request, code: str):
    """The event's analysis report as a PDF download (FR lxix)."""
    event = get_event_for(request.user, code)
    try:
        pdf = report.build(event, request.user)
    except Exception as exc:  # any failure becomes a readable message, not an error page (FR lxxvii)
        audit(Action.REPORT, request, target=event, ok=False, error=exc.__class__.__name__)
        messages.error(request, "The report could not be generated. Please try again; if it keeps failing, "
                                "tell an administrator (the error is in the audit trail).")
        return redirect("events:detail", code=code)
    audit(Action.REPORT, request, target=event, ok=True)
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{event.code}_report.pdf"'
    return response


@capability_required("administer")
def export_events(request, fmt: str):
    """The filtered event history as CSV or Excel (FR lxx)."""
    if fmt not in ("csv", "xlsx"):
        raise Http404()
    form = search.EventFilterForm(request.GET or None, user=request.user)
    events = search.apply(visible_events(request.user), form)
    stamp = timezone.localtime().strftime("%Y%m%d-%H%M")
    audit(Action.EXPORT, request, format=fmt, rows=events.count(), filters=form.active)
    if fmt == "csv":
        response = StreamingHttpResponse(export.csv_lines(events), content_type="text/csv; charset=utf-8")
    else:
        response = HttpResponse(export.xlsx_bytes(events),
                                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = f'attachment; filename="sonic_events_{stamp}.{fmt}"'
    return response


REVIEW_REASONS = [D.R_DISAGREE, D.R_LOW_CONF, D.R_POOR_QUALITY, D.R_SIMILAR, D.R_OVERLAP, D.R_UNSUPPORTED,
                  D.R_CRITICAL_NO_AGREEMENT, D.R_FALSE_ALARM, D.R_NO_GTM]


@capability_required("review")
def review_queue(request):
    waiting = review.queue()
    events = waiting
    filters = {"reason": request.GET.get("reason", ""), "severity": request.GET.get("severity", ""),
               "source": request.GET.get("source", "")}
    if filters["reason"] in REVIEW_REASONS:
        events = events.filter(review_reasons__icontains=filters["reason"])
    if filters["severity"]:
        events = events.filter(severity=filters["severity"])
    if filters["source"]:
        events = events.filter(record__source=filters["source"])
    page = Paginator(events, 25).get_page(request.GET.get("page"))
    return render(request, "events/review_queue.html", {
        "page": page, "filters": filters, "reasons": REVIEW_REASONS, "severities": list(reversed(SEVERITIES)),
        "sources": AudioRecord.Source.choices, "waiting": waiting.count(),
    })


def _after_review(request, event):
    """Back to the event, or on to the next one when the reviewer is working through the queue."""
    if request.POST.get("then") == "next":
        following = review.queue().exclude(pk=event.pk).first()
        if following:
            return redirect(f"{reverse('events:detail', args=[following.code])}?from=queue")
        messages.info(request, "The review queue is empty.")
        return redirect("events:review_queue")
    return redirect("events:detail", code=event.code)


@require_POST
@capability_required("review")
def review_decide(request, code: str):
    event = get_event_for(request.user, code)
    form = ReviewForm(request.POST, event=event)
    if not form.is_valid():
        messages.error(request, " ".join(e for errors in form.errors.values() for e in errors))
        return redirect("events:detail", code=code)
    try:
        review.decide(event, request.user, review.ReviewDecision(**form.cleaned_data), request=request)
    except review.ReviewRefused as exc:
        messages.error(request, str(exc))
        return redirect("events:detail", code=code)
    messages.success(request, f"{event.code}: {event.status.lower()} as {event.reviewed_class}.")
    return _after_review(request, event)


@require_POST
@capability_required("review")
def review_comment(request, code: str):
    event = get_event_for(request.user, code)
    form = CommentForm(request.POST)
    form.is_valid()
    try:
        review.comment(event, request.user, form.cleaned_data.get("comment", ""),
                       form.cleaned_data.get("recommended_action", ""), request=request)
    except review.ReviewRefused as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, "Comment added.")
    return redirect("events:detail", code=code)


@require_POST
@capability_required("review")
def review_status(request, code: str, change: str):
    if change not in ("close", "reopen"):
        raise Http404()
    event = get_event_for(request.user, code)
    try:
        (review.close if change == "close" else review.reopen)(event, request.user, request=request)
    except review.ReviewRefused as exc:
        messages.error(request, str(exc))
    else:
        messages.success(request, f"{event.code}: {event.status.lower()}.")
    return redirect("events:detail", code=code)
