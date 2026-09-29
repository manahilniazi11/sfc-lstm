from django.apps import AppConfig


class AccountsConfig(AppConfig):
    name = "sonic.web.accounts"
    label = "accounts"
    verbose_name = "Users and audit trail"

    def ready(self) -> None:
        from . import signals  # noqa: F401  (connects the login/logout audit handlers)
