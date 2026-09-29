"""Search and filtering of sound events (FR lxvii).

One form serves the history table, the timeline, the analytics page and the
data export, so "what you see is what you export". Category and severity
filter on the *effective* values: the reviewer's decision when there is one,
otherwise the automatic result.
"""

from __future__ import annotations

from django import forms
from django.db.models import F, Q

from sonic.detection import decision as D
from sonic.detection.rules import QUALITY_GRADES, SEVERITIES

from ..accounts.forms import StyledFormMixin
from ..accounts.models import User
from ..alerts.models import current_ruleset
from .models import AudioRecord

REVIEW_CHOICES = [
    ("", "Any review status"),
    ("waiting", "Waiting for review"),
    ("reviewed", "Reviewed (confirmed or corrected)"),
    ("corrected", "Corrected by a reviewer"),
    ("closed", "Closed"),
    ("not_needed", "No review needed"),
]


class DateInput(forms.DateInput):
    input_type = "date"


class EventFilterForm(StyledFormMixin, forms.Form):
    q = forms.CharField(label="Audio ID, event ID or file name", required=False)
    category = forms.ChoiceField(label="Sound category", required=False)
    severity = forms.ChoiceField(label="Severity", required=False)
    quality = forms.ChoiceField(label="Audio quality", required=False)
    review = forms.ChoiceField(label="Review status", required=False, choices=REVIEW_CHOICES)
    source = forms.ChoiceField(label="Source", required=False)
    user = forms.ChoiceField(label="User", required=False)
    date_from = forms.DateField(label="From", required=False, widget=DateInput)
    date_to = forms.DateField(label="To", required=False, widget=DateInput)
    conf_min = forms.IntegerField(label="Confidence from (%)", required=False, min_value=0, max_value=100)
    conf_max = forms.IntegerField(label="Confidence to (%)", required=False, min_value=0, max_value=100)

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        categories = sorted(current_ruleset().rules)
        self.fields["category"].choices = [("", "Any category")] + [(c, c) for c in categories]
        self.fields["severity"].choices = [("", "Any severity")] + [(s, s) for s in reversed(SEVERITIES)]
        self.fields["quality"].choices = [("", "Any quality")] + [(g, g) for g in QUALITY_GRADES]
        self.fields["source"].choices = [("", "Any source")] + list(AudioRecord.Source.choices)
        if user is not None and user.can("view_all_events"):
            people = User.objects.filter(audio_records__isnull=False).distinct().order_by("username")
            self.fields["user"].choices = [("", "Any user")] + [(u.user_code, f"{u.username} ({u.user_code})") for u in people]
        else:
            del self.fields["user"]  # a normal user only ever sees their own events

    def clean(self):
        data = super().clean()
        if data.get("date_from") and data.get("date_to") and data["date_from"] > data["date_to"]:
            self.add_error("date_to", "The end date is before the start date.")
        if data.get("conf_min") is not None and data.get("conf_max") is not None and data["conf_min"] > data["conf_max"]:
            self.add_error("conf_max", "The upper confidence is below the lower one.")
        return data

    @property
    def active(self) -> dict:
        """The filters that are set (for "n filters active" and for the export's audit entry)."""
        if not self.is_valid():
            return {}
        return {k: str(v) for k, v in self.cleaned_data.items() if v not in (None, "")}


def effective(reviewed: str, automatic: str, value: str) -> Q:
    """Match the reviewer's value when there is one, otherwise the automatic one."""
    return Q(**{reviewed: value}) | Q(**{reviewed: "", automatic: value})


def apply(events, form: EventFilterForm):
    """Narrow a SoundEvent queryset with the form's (valid) filters."""
    if not form.is_valid():
        return events
    f = form.cleaned_data
    if f["q"]:
        q = f["q"].strip()
        events = events.filter(Q(code__iexact=q) | Q(record__code__iexact=q) | Q(record__original_filename__icontains=q))
    if f["category"]:
        events = events.filter(effective("reviewed_class", "final_class", f["category"]))
    if f["severity"]:
        events = events.filter(effective("reviewed_severity", "severity", f["severity"]))
    if f["quality"]:
        events = events.filter(record__quality_grade=f["quality"])
    if f["source"]:
        events = events.filter(record__source=f["source"])
    if f.get("user"):
        events = events.filter(record__owner__user_code=f["user"])
    if f["date_from"]:
        events = events.filter(created_at__date__gte=f["date_from"])
    if f["date_to"]:
        events = events.filter(created_at__date__lte=f["date_to"])
    if f["conf_min"] is not None:
        events = events.filter(python_confidence__gte=f["conf_min"] / 100)
    if f["conf_max"] is not None:
        events = events.filter(python_confidence__lte=f["conf_max"] / 100)
    review = f["review"]
    if review == "waiting":
        events = events.filter(review_required=True).exclude(status__in=[D.REVIEWED, D.CLOSED])
    elif review == "reviewed":
        events = events.exclude(reviewed_class="")
    elif review == "corrected":
        events = events.exclude(reviewed_class="").exclude(reviewed_class=F("final_class"))
    elif review == "closed":
        events = events.filter(status=D.CLOSED)
    elif review == "not_needed":
        events = events.filter(review_required=False)
    return events

