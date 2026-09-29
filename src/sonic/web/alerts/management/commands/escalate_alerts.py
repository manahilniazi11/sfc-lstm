"""`python manage.py escalate_alerts`: escalate alerts nobody acknowledged in time.

The app also checks this while people use it; run the command from Windows
Task Scheduler or cron (e.g. every minute) so it happens with nobody logged in.
"""

from django.core.management.base import BaseCommand

from ...handling import escalate_overdue


class Command(BaseCommand):
    help = "Escalate active alerts that were not acknowledged within their rule's time."

    def handle(self, *args, **options):
        escalated = escalate_overdue()
        for alert in escalated:
            self.stdout.write(f"escalated {alert.code} ({alert.category}, {alert.severity})")
        self.stdout.write(self.style.SUCCESS(f"{len(escalated)} alert(s) escalated"))
