from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.templatetags.static import static
from django.urls import reverse
from django.views.decorators.http import require_POST

from ..accounts.decorators import capability_required
from ..alerts.models import DecisionSettings
from ..events import live
from ..events.models import MonitoringSession, SoundEvent
from ..events.pipeline import parse_gtm
from ..events.views import gtm_client_config


def _own_session(request, code: str) -> MonitoringSession:
    return get_object_or_404(MonitoringSession, code=code, user=request.user)


@capability_required("analyse")
def monitor(request):
    window = DecisionSettings.load().live_window_seconds
    recent = (SoundEvent.objects.filter(record__session__user=request.user).select_related("record")
              .order_by("-created_at")[:15])
    config = {**gtm_client_config(), "window_seconds": window, "start_url": reverse("monitoring:start"),
              "worklet_url": static("js/pcm-capture.js")}
    return render(request, "monitoring/monitor.html", {"config": config, "window_seconds": window, "recent": recent})


@require_POST
@capability_required("analyse")
def start(request):
    window = min(max(DecisionSettings.load().live_window_seconds, 1.0), 3.0)
    session = live.start_session(request.user, window, request.POST.get("device", ""), request=request)
    return JsonResponse({
        "session": session.code,
        "window_seconds": window,
        "window_url": reverse("monitoring:window", args=[session.code]),
        "stop_url": reverse("monitoring:stop", args=[session.code]),
    })


@require_POST
@capability_required("analyse")
def window(request, code: str):
    session = _own_session(request, code)
    if not session.active:
        return JsonResponse({"error": "this monitoring session has ended"}, status=409)
    upload = request.FILES.get("audio")
    try:
        start_s = float(request.POST.get("start", 0))
        result = live.analyse_window(session, upload.read() if upload else b"", parse_gtm(request.POST.get("gtm")),
                                     start_s, request=request)
    except (live.WindowRejected, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse(result)


@require_POST
@capability_required("analyse")
def stop(request, code: str):
    session = _own_session(request, code)
    live.stop_session(session, request=request)
    return JsonResponse({"session": session.code, "windows": session.windows_analysed})
