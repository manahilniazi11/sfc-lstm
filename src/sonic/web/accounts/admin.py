from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .models import AuditLog, User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ["user_code", "username", "email", "role", "is_active", "last_login"]
    list_filter = ["role", "is_active"]
    search_fields = ["user_code", "username", "email"]
    readonly_fields = ["user_code"]
    fieldsets = BaseUserAdmin.fieldsets + (
        ("SonicSentinel", {"fields": ["user_code", "role", "organisation", "job_title", "phone"]}),
    )


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ["created_at", "username", "action", "target_type", "target_id", "ip_address"]
    list_filter = ["action"]
    search_fields = ["username", "target_id"]

    # The audit trail is evidence: nobody edits or deletes it through the admin.
    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
