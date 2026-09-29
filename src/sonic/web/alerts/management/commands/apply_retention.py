"""`python manage.py apply_retention [--dry-run]`: apply the data-retention limits (see alerts/retention.py)."""

from django.core.management.base import BaseCommand

from ... import retention


class Command(BaseCommand):
    help = "Delete stored audio and event records older than the retention limits in Decision settings."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Only count what would be deleted.")

    def handle(self, *args, dry_run=False, **options):
        result = retention.plan() if dry_run else retention.apply()
        verb = "would delete" if dry_run else "deleted"
        self.stdout.write(f"{verb} {result.audio_files} audio file(s) and {result.event_records} event record(s); "
                          f"kept {result.kept_open} old record(s) that still need review or have an open alert")
