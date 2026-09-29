from django.urls import path

from . import views

app_name = "events"

urlpatterns = [
    path("upload/", views.upload, name="upload"),
    path("analyse/", views.analyse, name="analyse"),
    path("history/", views.history, name="history"),
    path("review/", views.review_queue, name="review_queue"),
    path("export.<str:fmt>", views.export_events, name="export"),
    path("<str:code>/", views.detail, name="detail"),
    path("<str:code>/audio/", views.audio, name="audio"),
    path("<str:code>/report.pdf", views.event_report, name="report"),
    path("<str:code>/review/", views.review_decide, name="review"),
    path("<str:code>/comment/", views.review_comment, name="comment"),
    path("<str:code>/status/<str:change>/", views.review_status, name="review_status"),
    path("<str:code>/<str:kind>.png", views.image, name="image"),
]
