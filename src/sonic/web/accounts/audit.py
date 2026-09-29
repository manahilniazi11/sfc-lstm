"""`record()` writes one audit-trail entry; every module calls it for the actions SRS 1.6 lxxvi lists."""

from __future__ import annotations

from .models import AuditLog

Action = AuditLog.Action


def client_ip(request) -> str | None:
    if request is None:
        return None
    return request.META.get("REMOTE_ADDR") or None


def record(action: str, request=None, user=None, target=None, **details) -> AuditLog:
    """Log `action` by `user` (default: the request's user) on `target` (any model instance)."""
    if user is None and request is not None and request.user.is_authenticated:
        user = request.user
    # A failed login has no user object, only the name that was typed.
    username = getattr(user, "username", "") or details.pop("username", "")
    return AuditLog.objects.create(
        user=user if user is not None and user.pk else None,
        username=username,
        action=action,
        target_type=type(target).__name__ if target is not None else "",
        target_id=str(getattr(target, "pk", "")) if target is not None else "",
        details=details,
        ip_address=client_ip(request),
    )
