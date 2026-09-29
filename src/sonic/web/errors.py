"""Understandable error pages (FR lxxvii) for failures that happen outside a view's own error handling.

A database failure (locked or missing SQLite file, disk full, ...) cannot use
the normal page layout, because the menu itself reads the database, so it
gets a standalone page with status 503 and the error is logged for the
administrator.
"""

from __future__ import annotations

import logging

from django.db import DatabaseError
from django.http import HttpResponse
from django.template.loader import render_to_string

log = logging.getLogger("sonic.web")


class DatabaseErrorMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        if isinstance(exception, DatabaseError):
            log.exception("database failure on %s", request.path, exc_info=exception)
            return HttpResponse(render_to_string("errors/database.html"), status=503)
        return None
