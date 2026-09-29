from django.urls import path

from . import settings_views as views

app_name = "settings"

urlpatterns = [
    path("", views.settings_page, name="decision"),
    path("retention/", views.run_retention, name="retention"),
    path("notices/", views.notices, name="notices"),
    path("notices/acknowledge/", views.acknowledge_notice, name="acknowledge_all"),
    path("notices/<int:pk>/acknowledge/", views.acknowledge_notice, name="acknowledge"),
    path("rules/", views.rule_list, name="rules"),
    path("rules/export.json", views.rules_export, name="rules_export"),
    path("rules/<str:category>/", views.rule_edit, name="rule"),
]
