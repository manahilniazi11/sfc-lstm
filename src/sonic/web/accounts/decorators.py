"""`@capability_required("review")` protects a view with the role matrix in roles.py."""

from __future__ import annotations

from functools import wraps

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied


def capability_required(capability: str):
    def decorator(view):
        @login_required
        @wraps(view)
        def wrapped(request, *args, **kwargs):
            if not request.user.can(capability):
                raise PermissionDenied(f"Your role cannot {capability.replace('_', ' ')}.")
            return view(request, *args, **kwargs)

        return wrapped

    return decorator
