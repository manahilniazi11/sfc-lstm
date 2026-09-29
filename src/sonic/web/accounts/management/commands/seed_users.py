"""Create one demo account per role for evaluators: `python manage.py seed_users`.

The shared password is read from SONIC_DEMO_PASSWORD (environment or .env),
so no credential is ever written into the source code. Existing accounts
keep their password unless --reset is given.
"""

from django.core.management.base import BaseCommand, CommandError

from sonic.dataset.config import setting
from sonic.web.accounts import roles
from sonic.web.accounts.models import User

DEMO_USERS = [
    ("admin", roles.ADMIN, "Ayesha", "Admin"),
    ("reviewer", roles.REVIEWER, "Rafay", "Reviewer"),
    ("security", roles.SECURITY, "Sana", "Security"),
    ("maintenance", roles.MAINTENANCE, "Moiz", "Maintenance"),
    ("user", roles.USER, "Usman", "User"),
]


class Command(BaseCommand):
    help = "Create demo accounts (one per role) with the password in SONIC_DEMO_PASSWORD."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="also reset the password of existing demo accounts")

    def handle(self, *args, reset=False, **options):
        password = setting("SONIC_DEMO_PASSWORD")
        if len(password) < 8:
            raise CommandError("Set SONIC_DEMO_PASSWORD (8+ characters) in the environment or .env first.")
        for username, role, first, last in DEMO_USERS:
            user, created = User.objects.get_or_create(
                username=username,
                defaults={"email": f"{username}@sonicsentinel.local", "role": role, "first_name": first, "last_name": last},
            )
            if created or reset:
                user.set_password(password)
                user.role = role
                user.is_staff = role == roles.ADMIN  # the admin may also open Django's admin site
                user.save()
            state = "created" if created else ("reset" if reset else "exists")
            self.stdout.write(f"{user.user_code}  {username:<12} {user.get_role_display():<22} {state}")
