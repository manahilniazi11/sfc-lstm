from __future__ import annotations

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

from . import roles
from .models import User


class StyledFormMixin:
    """Give every widget the app's form classes (form-control, form-select, form-check-input:
    see partials/tailwind.html) so templates can render fields in a loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            css = "form-select" if isinstance(field.widget, forms.Select) else "form-control"
            if isinstance(field.widget, forms.CheckboxInput):
                css = "form-check-input"
            field.widget.attrs.setdefault("class", css)


class LoginForm(StyledFormMixin, AuthenticationForm):
    pass


class RegisterForm(StyledFormMixin, UserCreationForm):
    """Self-registration always creates a Normal user; an administrator grants other roles.

    Letting people pick "Administrator" on the sign-up page would make the
    role system meaningless, so the requested role is only a note for the admin.
    """

    requested_role = forms.ChoiceField(
        choices=roles.CHOICES, initial=roles.USER, label="Role you need",
        help_text="An administrator approves roles other than Normal user.",
    )

    class Meta:
        model = User
        fields = ["username", "first_name", "last_name", "email", "organisation", "job_title"]

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email


class ProfileForm(StyledFormMixin, forms.ModelForm):
    class Meta:
        model = User
        fields = ["first_name", "last_name", "email", "organisation", "job_title", "phone"]

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exclude(pk=self.instance.pk).exists():
            raise forms.ValidationError("Another account uses this email.")
        return email


class RoleForm(StyledFormMixin, forms.ModelForm):
    class Meta:
        model = User
        fields = ["role", "is_active"]
