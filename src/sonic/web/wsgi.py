"""WSGI entry point for production servers (e.g. `waitress-serve sonic.web.wsgi:application`)."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sonic.web.settings")
application = get_wsgi_application()
