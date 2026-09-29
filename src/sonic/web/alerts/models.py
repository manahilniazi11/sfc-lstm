"""Rules and thresholds live in the database so administrators can change them in the app (FR xxxv, xxxvi, liii).

They are seeded from alert_rules/alert_rules.json the first time they are
needed, and `current_ruleset()` turns the rows back into the plain
dataclasses that sonic.detection uses.
"""

from __future__ import annotations

from django.conf import settings
from django.db import models

from sonic.detection import rules as R

from ..codes import CodedModel


class DecisionSettings(models.Model):
    """The single row of global thresholds and app settings (pk is always 1)."""

    min_confidence = models.FloatField(default=0.5, help_text="Below this a result is Low confidence and goes to manual review.")
    top_two_margin = models.FloatField(default=0.1, help_text="Minimum gap between the best and second-best class.")
    weak_match_difference = models.FloatField(default=0.3, help_text="Same class but confidences differ by more than this: Weak Match.")
    overlap_threshold = models.FloatField(default=0.25, help_text="Two event classes at or above this in one window: overlapping sounds.")
    unknown_confidence = models.FloatField(default=0.35, help_text="Below this the sound is reported as Unknown.")
    repeat_window_seconds = models.FloatField(default=10, help_text="Live: detections within this period count as repeated.")

    live_window_seconds = models.FloatField(default=2.0, help_text="Length of each live-microphone window (1-3 s).")
    max_upload_mb = models.PositiveIntegerField(default=25)
    audio_retention_days = models.PositiveIntegerField(default=90, help_text="Stored audio older than this is deleted.")
    event_retention_days = models.PositiveIntegerField(default=365, help_text="Event records older than this are deleted.")
    store_live_audio = models.BooleanField(default=True, help_text="Keep audio of live windows that raise an alert or need review.")

    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)

    THRESHOLD_FIELDS = ["min_confidence", "top_two_margin", "weak_match_difference", "overlap_threshold",
                        "unknown_confidence", "repeat_window_seconds"]

    class Meta:
        verbose_name_plural = "decision settings"

    @classmethod
    def load(cls) -> "DecisionSettings":
        obj = cls.objects.filter(pk=1).first()
        if obj is None:
            seeded = R.load_rules().thresholds
            obj = cls.objects.create(pk=1, **{f: getattr(seeded, f) for f in cls.THRESHOLD_FIELDS})
        return obj

    def thresholds(self) -> R.Thresholds:
        return R.Thresholds(**{f: getattr(self, f) for f in self.THRESHOLD_FIELDS})


class AlertRule(models.Model):
    """One category's rule; the fields mirror sonic.detection.rules.Rule."""

    SEVERITIES = [(s, s) for s in R.SEVERITIES]
    QUALITIES = [(q, q) for q in R.QUALITY_GRADES if q != "Unusable"]

    category = models.CharField(max_length=40, unique=True)
    enabled = models.BooleanField(default=True)
    critical = models.BooleanField(default=False)
    alert = models.BooleanField("raises an alert", default=False)
    severity = models.CharField(max_length=15, choices=SEVERITIES, default="Informational")
    alert_type = models.CharField(max_length=20, default="environment")
    min_confidence = models.FloatField(default=0.5)
    top_two_margin = models.FloatField(default=0.1)
    consecutive_detections = models.PositiveSmallIntegerField(default=1)
    require_model_agreement = models.BooleanField(default=False)
    min_quality = models.CharField(max_length=12, choices=QUALITIES, default="Poor")
    recommended_action = models.TextField(blank=True)
    manual_review = models.TextField("manual-review condition", blank=True)
    escalate_after_minutes = models.FloatField(null=True, blank=True, help_text="Unhandled alerts are escalated after this many minutes.")
    raise_to_critical = models.JSONField(null=True, blank=True)
    noise_limit_dbfs = models.FloatField(null=True, blank=True)
    above_noise_limit = models.JSONField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    RULE_FIELDS = ["category", "enabled", "critical", "alert", "severity", "alert_type", "min_confidence", "top_two_margin",
                   "consecutive_detections", "require_model_agreement", "min_quality", "recommended_action", "manual_review",
                   "escalate_after_minutes", "raise_to_critical", "noise_limit_dbfs", "above_noise_limit"]

    class Meta:
        ordering = ["-critical", "category"]

    def __str__(self) -> str:
        return self.category

    def to_rule(self) -> R.Rule:
        return R.Rule(**{f: getattr(self, f) for f in self.RULE_FIELDS})

    @classmethod
    def seed(cls) -> int:
        """Create rules from alert_rules.json for categories that have none yet."""
        created = 0
        for rule in R.load_rules().rules.values():
            _, new = cls.objects.get_or_create(category=rule.category, defaults={
                f: getattr(rule, f) for f in cls.RULE_FIELDS if f != "category"
            })
            created += new
        return created


def current_ruleset() -> R.RuleSet:
    if not AlertRule.objects.exists():
        AlertRule.seed()
    rules = {r.category: r.to_rule() for r in AlertRule.objects.all()}
    return R.RuleSet(DecisionSettings.load().thresholds(), rules)


class Alert(CodedModel):
    """An alert raised by a sound event (FR xlvi-xlix, liv-lvi)."""

    CODE_PREFIX = "ALR"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        ESCALATED = "escalated", "Escalated"
        DISMISSED = "dismissed", "Dismissed"
        RESOLVED = "resolved", "Resolved"

    OPEN = [Status.ACTIVE, Status.ESCALATED]  # nobody has taken it on yet: needs attention
    UNRESOLVED = OPEN + [Status.ACKNOWLEDGED]  # the incident is still going on

    event = models.ForeignKey("events.SoundEvent", on_delete=models.CASCADE, related_name="alerts")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    category = models.CharField(max_length=40)
    severity = models.CharField(max_length=15, db_index=True)
    alert_type = models.CharField(max_length=20)
    critical = models.BooleanField(default=False)
    message = models.CharField(max_length=255)
    recommended_action = models.TextField(blank=True)
    status = models.CharField(max_length=15, choices=Status.choices, default=Status.ACTIVE, db_index=True)
    escalate_after_minutes = models.FloatField(null=True, blank=True)
    handled_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    handled_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def is_open(self) -> bool:
        return self.status in self.OPEN


class AlertAction(models.Model):
    """Every change to an alert, by a person or by the automatic escalation (FR lvi)."""

    class Kind(models.TextChoices):
        CREATED = "created", "Created"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        DISMISSED = "dismissed", "Dismissed"
        ESCALATED = "escalated", "Escalated"
        AUTO_ESCALATED = "auto_escalated", "Escalated automatically"
        RESOLVED = "resolved", "Resolved"
        NOTE = "note", "Note"

    alert = models.ForeignKey(Alert, on_delete=models.CASCADE, related_name="actions")
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    kind = models.CharField(max_length=20, choices=Kind.choices)
    note = models.TextField(blank=True)

    class Meta:
        ordering = ["created_at", "id"]  # id: actions saved within the same instant stay in order


class AdminNotice(models.Model):
    """Something unusual administrators should know about (FR lxxviii); see alerts/anomalies.py."""

    class Kind(models.TextChoices):
        FAILED_UPLOADS = "failed_uploads", "Repeated failed uploads"
        MODEL_FAILURE = "model_failure", "Model failure"
        LOW_CONFIDENCE = "low_confidence", "Unusual low-confidence spike"
        CRITICAL_ALERTS = "critical_alerts", "Excessive critical alerts"
        DUPLICATES = "duplicates", "Duplicate files"
        FAILED_LOGINS = "failed_logins", "Failed login attempts"

    kind = models.CharField(max_length=20, choices=Kind.choices)
    key = models.CharField(max_length=150, blank=True)  # who or what it is about (a username, an IP, a model)
    message = models.CharField(max_length=300)
    count = models.PositiveIntegerField(default=1)
    first_seen = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now_add=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    acknowledged_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        ordering = ["-last_seen"]

    @property
    def is_open(self) -> bool:
        return self.acknowledged_at is None
