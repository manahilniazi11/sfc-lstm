from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import login, views as auth_views
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from . import roles
from .audit import Action, record
from .decorators import capability_required
from .forms import LoginForm, ProfileForm, RegisterForm, RoleForm
from .models import AuditLog, User


class LoginView(auth_views.LoginView):
    template_name = "accounts/login.html"
    authentication_form = LoginForm
    redirect_authenticated_user = True


def register(request):
    if request.user.is_authenticated:
        return redirect("home")
    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        record(Action.REGISTER, request, user=user, requested_role=form.cleaned_data["requested_role"])
        login(request, user)
        messages.success(request, f"Welcome! Your User ID is {user.user_code}.")
        return redirect("home")
    return render(request, "accounts/register.html", {"form": form})


@login_required
def profile(request):
    form = ProfileForm(request.POST or None, instance=request.user)
    if request.method == "POST" and form.is_valid():
        form.save()
        record(Action.PROFILE, request, target=request.user, fields=form.changed_data)
        messages.success(request, "Profile updated.")
        return redirect("accounts:profile")
    return render(request, "accounts/profile.html", {"form": form})


@capability_required("administer")
def user_list(request):
    users = User.objects.all()
    query = request.GET.get("q", "").strip()
    if query:
        users = users.filter(Q(username__icontains=query) | Q(user_code__icontains=query) | Q(email__icontains=query))
    registrations = {
        log.user_id: log.details.get("requested_role")
        for log in AuditLog.objects.filter(action=Action.REGISTER, user__in=users)
    }
    page = Paginator(users, 25).get_page(request.GET.get("page"))
    for user in page:
        user.requested_role = registrations.get(user.pk)
    return render(request, "accounts/user_list.html", {"page": page, "query": query, "role_choices": roles.CHOICES})


@require_POST
@capability_required("administer")
def user_update(request, pk: int):
    user = get_object_or_404(User, pk=pk)
    old = {"role": user.role, "is_active": user.is_active}
    form = RoleForm(request.POST, instance=user)
    if user == request.user:
        messages.error(request, "You cannot change your own role or deactivate yourself.")
    elif form.is_valid():
        form.save()
        record(Action.ROLE_CHANGE, request, target=user, before=old,
               after={"role": user.role, "is_active": user.is_active})
        messages.success(request, f"{user.user_code} updated.")
    return redirect("accounts:users")


@capability_required("administer")
def audit_trail(request):
    logs = AuditLog.objects.select_related("user")
    action = request.GET.get("action", "")
    if action:
        logs = logs.filter(action=action)
    page = Paginator(logs, 50).get_page(request.GET.get("page"))
    return render(request, "accounts/audit.html", {"page": page, "actions": AuditLog.Action.choices, "action": action})
