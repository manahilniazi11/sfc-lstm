"""Audio metadata and the sound events predicted from it (SRS Step 20, FR x, lxii, lxxii-lxxv)."""

from __future__ import annotations

from django.conf import settings
from django.db import models

from sonic.detection import decision as D

from ..codes import CodedModel


class MonitoringSession(CodedModel):
    """One live-microphone session: from Start to Stop (FR vi, lxxix)."""

    CODE_PREFIX = "MIC"
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="mic_sessions")
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    window_seconds = models.FloatField(default=2.0)
    windows_analysed = models.PositiveIntegerField(default=0)
    device_label = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ["-started_at"]

    @property
    def active(self) -> bool:
        return self.ended_at is None


class AudioRecord(CodedModel):
    """An uploaded file or a stored live window, with its metadata and quality (Audio ID = code)."""

    CODE_PREFIX = "AUD"

    class Source(models.TextChoices):
        UPLOAD = "upload", "Upload"
        BATCH = "batch", "Batch upload"
        LIVE = "live", "Live microphone"

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="audio_records")
    source = models.CharField(max_length=10, choices=Source.choices, default=Source.UPLOAD)
    session = models.ForeignKey(MonitoringSession, null=True, blank=True, on_delete=models.SET_NULL, related_name="records")
    uploaded_at = models.DateTimeField(auto_now_add=True, db_index=True)

    original_filename = models.CharField(max_length=255)
    stored_file = models.CharField(max_length=255, blank=True)  # relative to settings.UPLOAD_DIR; empty if not kept
    format = models.CharField(max_length=10)
    file_size = models.PositiveBigIntegerField()
    duration_s = models.FloatField()
    sample_rate = models.PositiveIntegerField()
    channels = models.PositiveSmallIntegerField()
    bit_depth = models.PositiveSmallIntegerField(null=True, blank=True)

    sha256 = models.CharField(max_length=64, db_index=True)
    fingerprint = models.BinaryField(null=True, blank=True, editable=False)
    duplicate_of = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="exact_copies")
    near_duplicate_of = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="near_copies")
    near_duplicate_score = models.FloatField(null=True, blank=True)

    quality_grade = models.CharField(max_length=12)
    quality_issues = models.JSONField(default=list, blank=True)
    quality_metrics = models.JSONField(default=dict, blank=True)

    # Evaluation only: when the file is byte-identical to a clip of our dataset,
    # its dataset label is kept for the model-comparison report. It is never
    # given to either model or to the decision.
    reference_audio_id = models.CharField(max_length=20, blank=True)
    reference_class = models.CharField(max_length=40, blank=True)
    reference_split = models.CharField(max_length=12, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of_id is not None or self.near_duplicate_of_id is not None


SEVERITY_CHOICES = [(s, s) for s in ["Informational", "Low", "Medium", "High", "Critical"]]
STATUS_CHOICES = [(s, s) for s in D.STATUSES]


class SoundEvent(CodedModel):
    """The analysis of one recording (or one live window): both models, their comparison and the decision."""

    CODE_PREFIX = "EVT"
    record = models.OneToOneField(AudioRecord, on_delete=models.CASCADE, related_name="event")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)  # processing timestamp
    segment_start_s = models.FloatField(default=0.0)
    segment_end_s = models.FloatField(default=0.0)
    processing_ms = models.FloatField(default=0.0)

    python_model_version = models.CharField(max_length=60)
    python_scores = models.JSONField(default=dict)
    python_class = models.CharField(max_length=40)
    python_confidence = models.FloatField()
    python_margin = models.FloatField()

    gtm_model_version = models.CharField(max_length=60, blank=True)
    gtm_scores = models.JSONField(null=True, blank=True)
    gtm_class = models.CharField(max_length=40, blank=True)
    gtm_confidence = models.FloatField(null=True, blank=True)
    gtm_margin = models.FloatField(null=True, blank=True)
    gtm_error = models.CharField(max_length=255, blank=True)

    class_match = models.BooleanField(default=False)
    confidence_difference = models.FloatField(null=True, blank=True)
    consistency = models.CharField(max_length=30)
    windows = models.JSONField(default=dict, blank=True)  # per-window scores of both models

    final_class = models.CharField(max_length=40, db_index=True)
    confidence_level = models.CharField(max_length=10)
    severity = models.CharField(max_length=15, choices=SEVERITY_CHOICES, db_index=True)
    alert_status = models.CharField(max_length=20)
    recommended_action = models.TextField(blank=True)
    review_required = models.BooleanField(default=False, db_index=True)
    review_reasons = models.JSONField(default=list, blank=True)
    overlapping_classes = models.JSONField(default=list, blank=True)
    repeated_detections = models.PositiveIntegerField(default=1)
    conditions = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=D.CLASSIFIED, db_index=True)

    # Set by a reviewer's decision; the model outputs above are never changed (FR lxi).
    reviewed_class = models.CharField(max_length=40, blank=True)
    reviewed_severity = models.CharField(max_length=15, choices=SEVERITY_CHOICES, blank=True)
    reviewed_action = models.TextField(blank=True)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                                    related_name="reviewed_events")
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def effective_class(self) -> str:
        """The class the app reports now: the reviewer's decision if there is one."""
        return self.reviewed_class or self.final_class

    @property
    def effective_severity(self) -> str:
        return self.reviewed_severity or self.severity

    @property
    def effective_action(self) -> str:
        return self.reviewed_action or self.recommended_action

    @property
    def awaiting_review(self) -> bool:
        return self.review_required and self.status not in (D.REVIEWED, D.CLOSED)

    def ranked(self, which: str = "python", n: int | None = None) -> list[tuple[str, float]]:
        scores = self.python_scores if which == "python" else (self.gtm_scores or {})
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:n] if n else ranked


class ReviewEntry(models.Model):
    """One reviewer action on an event: its complete review history (FR lix-lxi)."""

    class Kind(models.TextChoices):
        CONFIRM = "confirm", "Confirmed the class"
        CORRECT = "correct", "Corrected the class"
        COMMENT = "comment", "Comment"
        CLOSE = "close", "Closed"
        REOPEN = "reopen", "Reopened"

    event = models.ForeignKey(SoundEvent, on_delete=models.CASCADE, related_name="reviews")
    created_at = models.DateTimeField(auto_now_add=True)
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name="+")
    kind = models.CharField(max_length=10, choices=Kind.choices)
    previous_class = models.CharField(max_length=40, blank=True)
    new_class = models.CharField(max_length=40, blank=True)
    severity = models.CharField(max_length=15, blank=True)
    recommended_action = models.TextField(blank=True)
    comment = models.TextField(blank=True)

    class Meta:
        ordering = ["created_at", "id"]  # id: actions saved within the same instant stay in order
        verbose_name_plural = "review entries"
