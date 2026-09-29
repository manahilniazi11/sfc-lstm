from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("login/", views.LoginView.as_view(), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("register/", views.register, name="register"),
    path("profile/", views.profile, name="profile"),
    path("users/", views.user_list, name="users"),
    path("users/<int:pk>/", views.user_update, name="user_update"),
    path("audit/", views.audit_trail, name="audit"),
]
