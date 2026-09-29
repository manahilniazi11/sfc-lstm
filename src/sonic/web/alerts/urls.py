from django.urls import path

from . import views

app_name = "alerts"

urlpatterns = [
    path("", views.alert_list, name="list"),
    path("<str:code>/", views.alert_detail, name="detail"),
    path("<str:code>/<str:action>/", views.alert_action, name="action"),
]
