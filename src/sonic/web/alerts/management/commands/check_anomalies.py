"""`python manage.py check_anomalies`: raise administrator notices (see alerts/anomalies.py) from a scheduler."""

from django.core.management.base import BaseCommand

from ...anomalies import check


class Command(BaseCommand):
    help = "Look for failed uploads and logins, duplicates, alert floods, low-confidence spikes and model failures."

    def handle(self, *args, **options):
        raised = check()
        for notice in raised:
            self.stdout.write(f"{notice.get_kind_display()}: {notice.message}")
        self.stdout.write(self.style.SUCCESS(f"{len(raised)} notice(s) raised or updated"))
