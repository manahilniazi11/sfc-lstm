from __future__ import annotations

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models, transaction

from . import roles


class User(AbstractUser):
    """A registered person. `user_code` is the unique User ID shown in the app (SRS 1.6 iii)."""

    email = models.EmailField("email address", unique=True)
    user_code = models.CharField("User ID", max_length=12, unique=True, editable=False, blank=True)
    role = models.CharField(max_length=20, choices=roles.CHOICES, default=roles.USER)
    organisation = models.CharField(max_length=120, blank=True)
    job_title = models.CharField(max_length=120, blank=True)
    phone = models.CharField(max_length=30, blank=True)

    class Meta:
        ordering = ["user_code"]

    def save(self, *args, **kwargs):
        if self.is_superuser:
            self.role = roles.ADMIN
        if not self.user_code:
            with transaction.atomic():
                super().save(*args, **kwargs)  # the primary key numbers the User ID
                self.user_code = f"USR-{self.pk:05d}"
                super().save(update_fields=["user_code"])
            return
        super().save(*args, **kwargs)

    def can(self, capability: str) -> bool:
        return self.is_active and roles.can(self.role, capability)

    def __str__(self) -> str:
        return f"{self.user_code} {self.get_full_name() or self.username}"


class AuditLog(models.Model):
    """One recorded action (SRS 1.6 lxxvi): who did what, to which record, when and from where."""

    class Action(models.TextChoices):
        LOGIN = "login", "Login"
        LOGIN_FAILED = "login_failed", "Failed login"
        LOGOUT = "logout", "Logout"
        REGISTER = "register", "Registration"
        PROFILE = "profile", "Profile update"
        ROLE_CHANGE = "role_change", "Role change"
        UPLOAD = "upload", "Upload"
        UPLOAD_REJECTED = "upload_rejected", "Upload rejected"
        MIC_SESSION = "mic_session", "Microphone session"
        PREDICTION = "prediction", "Prediction"
        ALERT = "alert", "Alert"
        ALERT_ACTION = "alert_action", "Alert action"
        REVIEW = "review", "Review"
        OVERRIDE = "override", "Override"
        EXPORT = "export", "Export"
        REPORT = "report", "Report"
        SETTINGS = "settings", "Settings change"
        MODEL_UPDATE = "model_update", "Model update"
        RETENTION = "retention", "Retention clean-up"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    username = models.CharField(max_length=150, blank=True)  # kept even if the user is deleted
    action = models.CharField(max_length=30, choices=Action.choices, db_index=True)
    target_type = models.CharField(max_length=40, blank=True)
    target_id = models.CharField(max_length=40, blank=True)
    details = models.JSONField(default=dict, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.created_at:%Y-%m-%d %H:%M:%S} {self.username or '-'} {self.action}"
