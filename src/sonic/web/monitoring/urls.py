from django.urls import path

from . import views

app_name = "monitoring"

urlpatterns = [
    path("", views.monitor, name="monitor"),
    path("start/", views.start, name="start"),
    path("<str:code>/window/", views.window, name="window"),
    path("<str:code>/stop/", views.stop, name="stop"),
]
