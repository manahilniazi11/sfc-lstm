"""Audit logins, logouts and failed logins (failed logins also feed the anomaly alerts)."""

from django.contrib.auth.signals import user_logged_in, user_logged_out, user_login_failed
from django.dispatch import receiver

from .audit import Action, record


@receiver(user_logged_in)
def on_login(sender, request, user, **kwargs):
    record(Action.LOGIN, request, user=user)


@receiver(user_logged_out)
def on_logout(sender, request, user, **kwargs):
    if user is not None:
        record(Action.LOGOUT, request, user=user)


@receiver(user_login_failed)
def on_login_failed(sender, credentials, request=None, **kwargs):
    # Never store the attempted password; the username is enough to spot attacks.
    record(Action.LOGIN_FAILED, request, username=credentials.get("username", ""))
