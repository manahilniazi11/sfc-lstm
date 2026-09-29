"""Accounts: registration, User IDs, roles, access control and the audit trail."""

import pytest
from django.urls import reverse

from sonic.web.accounts import roles
from sonic.web.accounts.models import AuditLog, User

PASSWORD = "Sonic-test-2026"


def make_user(username, role=roles.USER, **extra):
    return User.objects.create_user(username=username, email=f"{username}@example.com", password=PASSWORD, role=role, **extra)


@pytest.mark.django_db
def test_every_user_gets_a_unique_user_id():
    a, b = make_user("alice"), make_user("bob")
    assert a.user_code.startswith("USR-") and b.user_code.startswith("USR-")
    assert a.user_code != b.user_code


@pytest.mark.django_db
def test_registration_creates_a_normal_user_even_if_admin_is_requested(client):
    response = client.post(reverse("accounts:register"), {
        "username": "carol", "first_name": "Carol", "last_name": "K", "email": "carol@example.com",
        "organisation": "Plant 1", "job_title": "Engineer", "requested_role": roles.ADMIN,
        "password1": PASSWORD, "password2": PASSWORD,
    })
    assert response.status_code == 302
    user = User.objects.get(username="carol")
    assert user.role == roles.USER  # a sign-up form must never grant admin rights
    log = AuditLog.objects.get(action=AuditLog.Action.REGISTER)
    assert log.details["requested_role"] == roles.ADMIN


@pytest.mark.django_db
def test_duplicate_email_is_rejected(client):
    make_user("dave")
    response = client.post(reverse("accounts:register"), {
        "username": "dave2", "email": "DAVE@example.com", "requested_role": roles.USER,
        "password1": PASSWORD, "password2": PASSWORD,
    })
    assert response.status_code == 200
    assert not User.objects.filter(username="dave2").exists()


@pytest.mark.django_db
def test_login_and_failed_login_are_audited(client):
    make_user("erin")
    assert not client.login(username="erin", password="wrong-password")
    assert client.login(username="erin", password=PASSWORD)
    actions = set(AuditLog.objects.values_list("action", flat=True))
    assert {AuditLog.Action.LOGIN, AuditLog.Action.LOGIN_FAILED} <= actions
    failed = AuditLog.objects.get(action=AuditLog.Action.LOGIN_FAILED)
    assert failed.username == "erin" and "password" not in failed.details


@pytest.mark.django_db
def test_pages_require_login(client):
    response = client.get(reverse("home"))
    assert response.status_code == 302 and reverse("accounts:login") in response.url


@pytest.mark.django_db
@pytest.mark.parametrize("role,allowed", [(roles.USER, False), (roles.REVIEWER, False), (roles.ADMIN, True)])
def test_only_admins_manage_users(client, role, allowed):
    make_user("frank", role=role)
    client.login(username="frank", password=PASSWORD)
    response = client.get(reverse("accounts:users"))
    assert response.status_code == (200 if allowed else 403)


@pytest.mark.django_db
def test_admin_changes_a_role_and_it_is_audited(client):
    make_user("root", role=roles.ADMIN)
    target = make_user("gina")
    client.login(username="root", password=PASSWORD)
    client.post(reverse("accounts:user_update", args=[target.pk]), {"role": roles.REVIEWER, "is_active": "on"})
    target.refresh_from_db()
    assert target.role == roles.REVIEWER
    log = AuditLog.objects.get(action=AuditLog.Action.ROLE_CHANGE)
    assert log.details["before"]["role"] == roles.USER and log.details["after"]["role"] == roles.REVIEWER


@pytest.mark.django_db
def test_admin_cannot_demote_themselves(client):
    admin = make_user("root", role=roles.ADMIN)
    client.login(username="root", password=PASSWORD)
    client.post(reverse("accounts:user_update", args=[admin.pk]), {"role": roles.USER})
    admin.refresh_from_db()
    assert admin.role == roles.ADMIN


def test_capability_matrix():
    assert roles.can(roles.USER, "analyse") and not roles.can(roles.USER, "batch_upload")
    assert roles.can(roles.REVIEWER, "review") and not roles.can(roles.SECURITY, "review")
    assert roles.can(roles.SECURITY, "handle_alerts") and roles.can(roles.MAINTENANCE, "handle_alerts")
    assert all(roles.can(roles.ADMIN, cap) for cap in roles.CAPABILITIES)
    with pytest.raises(KeyError):
        roles.can(roles.ADMIN, "fly")


@pytest.mark.django_db
def test_superuser_is_always_admin():
    user = User.objects.create_superuser(username="su", email="su@example.com", password=PASSWORD)
    assert user.role == roles.ADMIN and user.can("administer")


@pytest.mark.django_db
def test_inactive_user_has_no_capabilities():
    user = make_user("hank", role=roles.ADMIN, is_active=False)
    assert not user.can("administer")
