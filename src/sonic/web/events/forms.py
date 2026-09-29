from __future__ import annotations

from django import forms

from sonic.detection.rules import SEVERITIES, UNKNOWN

from ..accounts.forms import StyledFormMixin
from .review import class_choices


class ReviewForm(StyledFormMixin, forms.Form):
    decision = forms.ChoiceField(choices=[("confirm", "Confirm the detected class"), ("correct", "Correct it")],
                                 widget=forms.RadioSelect, initial="confirm")
    category = forms.ChoiceField(label="Correct class", required=False)
    severity = forms.ChoiceField(label="Severity", required=False,
                                 help_text="Automatic keeps the result's severity (or the corrected class's).")
    recommended_action = forms.CharField(label="Recommended action", required=False,
                                         widget=forms.Textarea(attrs={"rows": 2}),
                                         help_text="Leave empty to keep the automatic recommendation.")
    comment = forms.CharField(label="Comment", required=False, widget=forms.Textarea(attrs={"rows": 3}))
    close = forms.BooleanField(label="Close the event (nothing more to do)", required=False)

    def __init__(self, *args, event=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["category"].choices = [("", "—")] + [(c, c) for c in class_choices()]
        self.fields["severity"].choices = [("", "Automatic")] + [(s, s) for s in reversed(SEVERITIES)]
        self.fields["decision"].widget.attrs["class"] = "form-check-input"
        if event is not None and event.final_class == UNKNOWN:
            self.initial["decision"] = "correct"

    def clean(self):
        data = super().clean()
        if data.get("decision") == "correct" and not data.get("category"):
            self.add_error("category", "Choose the class you heard.")
        return data


class CommentForm(StyledFormMixin, forms.Form):
    comment = forms.CharField(label="Comment", required=False, widget=forms.Textarea(attrs={"rows": 2}))
    recommended_action = forms.CharField(label="Recommended action", required=False,
                                         widget=forms.Textarea(attrs={"rows": 2}))
