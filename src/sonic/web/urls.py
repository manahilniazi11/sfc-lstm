from django.contrib import admin
from django.urls import include, path

from .events import dashboard

urlpatterns = [
    path("", dashboard.home, name="home"),
    path("analytics/", dashboard.analytics, name="analytics"),
    path("accounts/", include("sonic.web.accounts.urls")),
    path("events/", include("sonic.web.events.urls")),
    path("monitor/", include("sonic.web.monitoring.urls")),
    path("alerts/", include("sonic.web.alerts.urls")),
    path("settings/", include("sonic.web.alerts.settings_urls")),
    path("django-admin/", admin.site.urls),
]
