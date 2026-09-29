from __future__ import annotations

from django import forms

from sonic.detection import rules as R

from ..accounts.forms import StyledFormMixin
from .models import AlertRule, DecisionSettings

PROBABILITY = {"min_value": 0.0, "max_value": 1.0}


class DecisionSettingsForm(StyledFormMixin, forms.ModelForm):
    """Global thresholds (FR xxxv, xxxvi), live window, upload limit and retention (FR lxxx)."""

    class Meta:
        model = DecisionSettings
        fields = ["min_confidence", "top_two_margin", "unknown_confidence", "weak_match_difference",
                  "overlap_threshold", "repeat_window_seconds", "live_window_seconds", "max_upload_mb",
                  "audio_retention_days", "event_retention_days", "store_live_audio"]
        widgets = {f: forms.NumberInput(attrs={"step": "0.01"}) for f in
                   ["min_confidence", "top_two_margin", "unknown_confidence", "weak_match_difference", "overlap_threshold"]}

    def clean(self):
        data = super().clean()
        for name in ["min_confidence", "top_two_margin", "unknown_confidence", "weak_match_difference", "overlap_threshold"]:
            value = data.get(name)
            if value is not None and not 0 <= value <= 1:
                self.add_error(name, "Use a value between 0 and 1 (e.g. 0.5 for 50%).")
        if data.get("unknown_confidence") is not None and data.get("min_confidence") is not None \
                and data["unknown_confidence"] > data["min_confidence"]:
            self.add_error("unknown_confidence", "Must not be above the minimum confidence: below the minimum a result "
                                                 "is uncertain, below this lower limit it becomes Unknown.")
        window = data.get("live_window_seconds")
        if window is not None and not 1 <= window <= 3:
            self.add_error("live_window_seconds", "Live windows must be 1 to 3 seconds long.")
        if data.get("repeat_window_seconds") is not None and data["repeat_window_seconds"] <= 0:
            self.add_error("repeat_window_seconds", "Must be more than 0 seconds.")
        if data.get("max_upload_mb") is not None and not 1 <= data["max_upload_mb"] <= 200:
            self.add_error("max_upload_mb", "Use 1 to 200 MB.")
        audio, events = data.get("audio_retention_days"), data.get("event_retention_days")
        if audio is not None and audio < 1:
            self.add_error("audio_retention_days", "Keep audio at least 1 day.")
        if audio is not None and events is not None and audio > events:
            self.add_error("audio_retention_days", "Audio cannot be kept longer than its event record.")
        return data


class AlertRuleForm(StyledFormMixin, forms.ModelForm):
    """One category's alert rule (FR liii): the fields mirror sonic.detection.rules.Rule."""

    class Meta:
        model = AlertRule
        fields = ["enabled", "critical", "alert", "severity", "alert_type", "min_confidence", "top_two_margin",
                  "consecutive_detections", "require_model_agreement", "min_quality", "recommended_action",
                  "manual_review", "escalate_after_minutes", "raise_to_critical", "noise_limit_dbfs",
                  "above_noise_limit"]
        widgets = {
            "min_confidence": forms.NumberInput(attrs={"step": "0.01"}),
            "top_two_margin": forms.NumberInput(attrs={"step": "0.01"}),
            "recommended_action": forms.Textarea(attrs={"rows": 2}),
            "manual_review": forms.Textarea(attrs={"rows": 2}),
            "raise_to_critical": forms.Textarea(attrs={"rows": 2}),
            "above_noise_limit": forms.Textarea(attrs={"rows": 2}),
        }
        help_texts = {
            "min_confidence": "Python confidence (0-1) needed before this class raises an alert.",
            "top_two_margin": "Gap (0-1) needed between the best and second-best class.",
            "consecutive_detections": "Windows in a row (live: repeats within the repeat period) needed to confirm.",
            "min_quality": "The worst audio quality that may still raise an alert.",
            "raise_to_critical": 'Optional, e.g. {"min_confidence": 0.85, "consecutive_detections": 3}.',
            "noise_limit_dbfs": "Background Noise only: louder than this (dBFS) is treated as a disturbance.",
            "above_noise_limit": 'Background Noise only, e.g. {"severity": "Medium", "alert": true}.',
        }

    def clean(self):
        data = super().clean()
        if self.errors:
            return data
        try:  # the detection code's own checks decide what a valid rule is
            R.rule_from_dict({"category": self.instance.category, **data})
        except (TypeError, ValueError) as exc:
            raise forms.ValidationError(str(exc))
        for name in ["raise_to_critical", "above_noise_limit"]:
            if data.get(name) is not None and not isinstance(data[name], dict):
                self.add_error(name, 'Write a JSON object such as {"min_confidence": 0.85}, or leave it empty.')
        if data.get("escalate_after_minutes") is not None and data["escalate_after_minutes"] <= 0:
            self.add_error("escalate_after_minutes", "Must be more than 0 minutes (or empty for no escalation).")
        return data
