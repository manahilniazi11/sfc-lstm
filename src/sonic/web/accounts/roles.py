"""The five SRS roles and what each may do (SRS 1.6 ii, role-based access control).

Permissions are grouped into *capabilities*. Views ask "may this user
`review`?" instead of checking role names, so the matrix below is the one
place to read or change who can do what.
"""

from __future__ import annotations

USER = "user"
REVIEWER = "reviewer"
SECURITY = "security"
MAINTENANCE = "maintenance"
ADMIN = "admin"

CHOICES = [
    (USER, "Normal user"),
    (REVIEWER, "Audio reviewer"),
    (SECURITY, "Security operator"),
    (MAINTENANCE, "Maintenance operator"),
    (ADMIN, "Administrator"),
]

OPERATORS = {REVIEWER, SECURITY, MAINTENANCE, ADMIN}

CAPABILITIES: dict[str, set[str]] = {
    # upload one clip, run live monitoring, see own events and reports
    "analyse": {USER, REVIEWER, SECURITY, MAINTENANCE, ADMIN},
    # several clips at once (SRS 1.6 v: "authorized users")
    "batch_upload": OPERATORS,
    # every user's events, the timeline and search
    "view_all_events": OPERATORS,
    # acknowledge / dismiss / escalate alerts
    "handle_alerts": {SECURITY, MAINTENANCE, ADMIN},
    # manual-review queue: listen, confirm or correct, comment, override
    "review": {REVIEWER, ADMIN},
    # analytics and the administrator dashboard
    "analytics": {REVIEWER, ADMIN},
    # users, roles, thresholds, alert rules, retention, data export, audit trail
    "administer": {ADMIN},
}


def can(role: str, capability: str) -> bool:
    if capability not in CAPABILITIES:
        raise KeyError(f"unknown capability {capability!r}")
    return role in CAPABILITIES[capability]
