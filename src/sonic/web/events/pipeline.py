"""Everything that happens to one recording, in order (SRS Steps 3-19).

1. validate the file (format, size, integrity, duration, sample rate, channels, usable sound)
2. extract metadata and assess audio quality
3. check for exact and near duplicates
4. run the Python model on the server
5. take the GTM scores that the browser computed on its own
6. compare the two and make the final decision with the alert rules
7. store the record, the event, the images and any alert; write the audit trail

The GTM model runs in the browser (TensorFlow.js, the export format Teachable
Machine supports) *before* the file is sent, so it never sees the Python
result; the server only receives its finished scores.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.db import transaction

from sonic.dataset.config import DEFAULT_METADATA_DIR
from sonic.detection import decision as D
from sonic.detection.compare import Scores
from sonic.detection.rules import NON_EVENTS
from sonic.inference import audio_io, fingerprint, visuals
from sonic.inference.python_model import get_classifier
from sonic.quality.analyze import assess

from ..accounts.audit import Action, record as audit
from ..alerts.anomalies import report_model_failure
from ..alerts.models import Alert, AlertAction, current_ruleset
from .models import AudioRecord, SoundEvent

MIN_SAMPLE_RATE = 8000
MAX_CHANNELS = 8
MAX_DURATION_S = 600
NEAR_DUPLICATE_CANDIDATES = 300


MODEL_FAILURE_MESSAGE = ("The Python model could not analyse this recording. Please try again in a few minutes; "
                         "the administrators have been notified.")


class UploadRejected(Exception):
    """The recording failed validation; the message is shown to the user."""


@dataclass
class GtmResult:
    version: str = ""
    scores: dict[str, float] | None = None
    windows: list[dict] = field(default_factory=list)
    error: str = ""
    elapsed_ms: float | None = None  # time GTM took in the browser


def parse_gtm(payload) -> GtmResult:
    """Validate the browser's GTM result. Anything malformed counts as "GTM unavailable"."""
    if not payload:
        return GtmResult(error="no GTM result was sent")
    try:
        data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        if data.get("error"):
            return GtmResult(version=str(data.get("version", ""))[:60], error=str(data["error"])[:250])
        labels, clip = data["labels"], data["clip"]
        if not (isinstance(labels, list) and isinstance(clip, list) and len(labels) == len(clip) >= 2):
            raise ValueError("labels and scores differ in length")
        if any(not isinstance(label, str) or len(label) > 40 for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("invalid labels")

        def probs(values):
            values = [float(v) for v in values]
            if any(not math.isfinite(v) or v < -1e-6 or v > 1 + 1e-6 for v in values) or abs(sum(values) - 1) > 0.02:
                raise ValueError("scores are not probabilities")
            return {label: round(min(max(v, 0.0), 1.0), 5) for label, v in zip(labels, values)}

        windows = [
            {"start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3), "scores": probs(w["probs"])}
            for w in data.get("windows", [])[:2000]
        ]
        elapsed = data.get("elapsed_ms")
        elapsed = round(float(elapsed), 1) if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0 else None
        return GtmResult(version=str(data.get("version", ""))[:60], scores=probs(clip), windows=windows, elapsed_ms=elapsed)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return GtmResult(error=f"invalid GTM result ({exc})")


@lru_cache(maxsize=1)
def _reference_index() -> dict[str, tuple[str, str, str]]:
    """sha256 -> (audio ID, class, split) of the dataset clips, for evaluation only."""
    path = DEFAULT_METADATA_DIR / "dataset_metadata.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["sha256"]: (r["audio_id"], r["class_name"], r["split"]) for r in csv.DictReader(f) if r.get("sha256")}


def save_upload(uploaded, max_mb: int) -> tuple[Path, str, int]:
    """Stream an uploaded file to a temporary path while hashing it; checks format and size first."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in audio_io.SUPPORTED_FORMATS:
        raise UploadRejected(f"Unsupported format '{suffix or 'none'}'. Use WAV, MP3, FLAC, OGG or M4A.")
    if uploaded.size > max_mb * 1024 * 1024:
        raise UploadRejected(f"The file is {uploaded.size / 1048576:.1f} MB; the limit is {max_mb} MB.")
    if uploaded.size == 0:
        raise UploadRejected("The file is empty.")
    tmp_dir = Path(settings.UPLOAD_DIR) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=suffix, delete=False) as out:
        for chunk in uploaded.chunks():
            digest.update(chunk)
            out.write(chunk)
    return Path(out.name), digest.hexdigest(), uploaded.size


def _validate(path: Path):
    try:
        audio, sr, info = audio_io.decode(path)
    except audio_io.AudioDecodeError as exc:
        raise UploadRejected(f"The audio could not be decoded: {exc}.") from exc
    if info.sample_rate < MIN_SAMPLE_RATE:
        raise UploadRejected(f"Sampling rate {info.sample_rate} Hz is too low (minimum {MIN_SAMPLE_RATE} Hz).")
    if info.channels > MAX_CHANNELS:
        raise UploadRejected(f"{info.channels} channels is not supported (maximum {MAX_CHANNELS}).")
    if info.duration_s > MAX_DURATION_S:
        raise UploadRejected(f"The recording is {info.duration_s / 60:.1f} minutes long; the limit is {MAX_DURATION_S // 60} minutes.")
    quality = assess(audio, sr)
    if not quality.usable:
        raise UploadRejected("The recording is unusable: " + "; ".join(quality.issues) + ".")
    return audio, sr, info, quality


def _event_marks(times, window_scores: list[dict], min_conf: float) -> list[tuple[float, float, str]]:
    """Spans of consecutive windows where the Python model heard an event (for the waveform)."""
    marks: list[list] = []
    for (start, end), scores in zip(times, window_scores):
        top = max(scores, key=scores.get)
        if top in NON_EVENTS or scores[top] < min_conf:
            continue
        if marks and marks[-1][2] == top and start <= marks[-1][1]:
            marks[-1][1] = end
        else:
            marks.append([start, end, top])
    return [tuple(m) for m in marks]


def _find_near_duplicate(record: AudioRecord, fp) -> tuple[AudioRecord | None, float | None]:
    if len(fp) < fingerprint.MIN_FRAMES:
        return None, None
    candidates = (AudioRecord.objects.exclude(pk=record.pk).exclude(fingerprint=None)
                  .exclude(sha256=record.sha256).order_by("-uploaded_at")[:NEAR_DUPLICATE_CANDIDATES])
    best, best_score = None, 0.0
    for other in candidates:
        score = fingerprint.similarity(fp, fingerprint.from_bytes(bytes(other.fingerprint)))
        if score > best_score:
            best, best_score = other, score
    if best is not None and best_score >= fingerprint.THRESHOLD:
        # Point at the first recording, not at another copy of it.
        original = best.duplicate_of or best.near_duplicate_of or best
        return original, round(best_score, 4)
    return None, None


def analyse_file(
    path: Path,
    *,
    original_name: str,
    sha256: str,
    size: int,
    user,
    gtm: GtmResult,
    source: str = AudioRecord.Source.UPLOAD,
    request=None,
) -> SoundEvent:
    """Validate and analyse an uploaded file already saved at `path` (the file is moved or deleted)."""
    try:
        audio, sr, info, quality = _validate(path)
    except UploadRejected as exc:
        path.unlink(missing_ok=True)
        audit(Action.UPLOAD_REJECTED, request, user=user, filename=original_name, reason=str(exc))
        raise

    # The spectrogram and the fingerprint do not depend on the model, so they are
    # computed on a second thread while the Python model runs (TensorFlow and
    # NumPy release Python's lock while they compute).
    with ThreadPoolExecutor(max_workers=1) as pool:
        side = pool.submit(lambda: (visuals.spectrogram_png(audio, sr), fingerprint.fingerprint(audio, sr)))
        try:
            python = get_classifier().analyse(audio, sr)
        except Exception as exc:  # a model failure is reported, not shown as an error page (FR lxxvii, lxxviii)
            path.unlink(missing_ok=True)
            report_model_failure("Python model", exc, f"the upload {original_name}")
            audit(Action.UPLOAD_REJECTED, request, user=user, filename=original_name,
                  reason=f"model failure ({exc.__class__.__name__})")
            raise UploadRejected(MODEL_FAILURE_MESSAGE) from exc
        spectrogram, fp = side.result()
    if python.clip_probs is None:
        path.unlink(missing_ok=True)
        reason = "Every one-second window is below the silence level; there is no sound to classify."
        audit(Action.UPLOAD_REJECTED, request, user=user, filename=original_name, reason=reason)
        raise UploadRejected(reason)

    ruleset = current_ruleset()
    window_scores = python.window_scores()
    decision = D.decide(
        Scores(python.clip_scores()),
        Scores(gtm.scores) if gtm.scores else None,
        quality.grade,
        ruleset,
        windows=[Scores(w) for w in window_scores],
        noise_level_dbfs=quality.metrics.get("active_level_dbfs"),
    )
    reference = _reference_index().get(sha256)

    with transaction.atomic():
        record = AudioRecord.objects.create(
            owner=user, source=source, original_filename=original_name[:255], format=info.format,
            file_size=size, duration_s=info.duration_s, sample_rate=info.sample_rate, channels=info.channels,
            bit_depth=info.bit_depth, sha256=sha256, fingerprint=fingerprint.to_bytes(fp),
            quality_grade=quality.grade, quality_issues=quality.issues, quality_metrics=quality.metrics,
            reference_audio_id=reference[0] if reference else "", reference_class=reference[1] if reference else "",
            reference_split=reference[2] if reference else "",
        )
        record.duplicate_of = AudioRecord.objects.filter(sha256=sha256).exclude(pk=record.pk).order_by("uploaded_at").first()
        if record.duplicate_of is None:
            record.near_duplicate_of, record.near_duplicate_score = _find_near_duplicate(record, fp)

        folder = Path(settings.UPLOAD_DIR) / record.code
        folder.mkdir(parents=True, exist_ok=True)
        stored = folder / f"original{path.suffix.lower()}"
        shutil.move(str(path), stored)
        record.stored_file = f"{record.code}/{stored.name}"
        record.save()

        marks = _event_marks(python.window_times, window_scores, ruleset.thresholds.min_confidence)
        (folder / "waveform.png").write_bytes(visuals.waveform_png(audio, sr, marks))
        (folder / "spectrogram.png").write_bytes(spectrogram)

        event = _create_event(record, decision, python, window_scores, gtm, 0.0, info.duration_s)
        audit(Action.UPLOAD, request, user=user, target=record, filename=original_name, source=source,
              duplicate_of=str(record.duplicate_of or ""), near_duplicate_of=str(record.near_duplicate_of or ""))
        audit(Action.PREDICTION, request, user=user, target=event, python=event.python_class, gtm=event.gtm_class,
              final=event.final_class, python_model=event.python_model_version, gtm_model=event.gtm_model_version)
        if decision.alert:
            raise_alert(event, ruleset.for_category(decision.final_class), request=request, user=user)
    return event


def _create_event(record, decision, python, window_scores, gtm: GtmResult, start_s: float, end_s: float) -> SoundEvent:
    cmp = decision.comparison
    return SoundEvent.objects.create(
        record=record,
        segment_start_s=round(start_s, 3), segment_end_s=round(end_s, 3), processing_ms=round(python.elapsed_ms, 1),
        python_model_version=python.version, python_scores=python.clip_scores(),
        python_class=cmp.python_class, python_confidence=round(cmp.python_confidence, 5), python_margin=round(cmp.python_margin, 5),
        gtm_model_version=gtm.version, gtm_scores=gtm.scores, gtm_class=cmp.gtm_class or "",
        gtm_confidence=cmp.gtm_confidence, gtm_margin=cmp.gtm_margin, gtm_error=gtm.error,
        class_match=cmp.match, confidence_difference=cmp.confidence_difference, consistency=cmp.status,
        windows={
            "python": [{"start": s, "end": e, "scores": w} for (s, e), w in zip(python.window_times, window_scores)],
            "gtm": gtm.windows,
            "preprocessing": python.preprocessing or {},
            "gtm_ms": gtm.elapsed_ms,
        },
        final_class=decision.final_class, confidence_level=decision.confidence_level, severity=decision.severity,
        alert_status=decision.alert_status, recommended_action=decision.recommended_action,
        review_required=decision.review_required, review_reasons=decision.review_reasons,
        overlapping_classes=decision.overlapping_classes, repeated_detections=decision.repeated_detections,
        conditions=decision.conditions, status=decision.status,
    )


def raise_alert(event: SoundEvent, rule, request=None, user=None) -> Alert:
    alert = Alert.objects.create(
        event=event, category=event.final_class, severity=event.severity, alert_type=rule.alert_type,
        critical=rule.critical, message=f"{event.final_class} detected ({event.python_confidence:.0%} confidence)",
        recommended_action=event.recommended_action, escalate_after_minutes=rule.escalate_after_minutes,
    )
    AlertAction.objects.create(alert=alert, kind=AlertAction.Kind.CREATED, note=alert.message)
    audit(Action.ALERT, request, user=user, target=alert, event=event.code, severity=alert.severity, category=alert.category)
    return alert
