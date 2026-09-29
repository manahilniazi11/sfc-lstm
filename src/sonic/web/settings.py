"""Django settings for the SonicSentinel web app.

Machine-specific values (secret key, debug flag, where uploads are kept)
come from environment variables or the repository's .env file, the same way
the data pipeline reads SONIC_DATA_DIR.
"""

from __future__ import annotations

from pathlib import Path

from ..dataset.config import AUDIO_DATA_DIR, REPO_ROOT, setting

BASE_DIR = REPO_ROOT

DEBUG = setting("SONIC_DEBUG", "1") == "1"
SECRET_KEY = setting("SONIC_SECRET_KEY", "dev-only-insecure-key-change-me" if DEBUG else "")
if not SECRET_KEY:
    raise RuntimeError("Set SONIC_SECRET_KEY in the environment or .env when SONIC_DEBUG=0")
ALLOWED_HOSTS = [h.strip() for h in setting("SONIC_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]
CSRF_TRUSTED_ORIGINS = [o.strip() for o in setting("SONIC_CSRF_TRUSTED_ORIGINS", "").split(",") if o.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "sonic.web.accounts",
    "sonic.web.events",
    "sonic.web.alerts",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "sonic.web.errors.DatabaseErrorMiddleware",
]

ROOT_URLCONF = "sonic.web.urls"
WSGI_APPLICATION = "sonic.web.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "sonic.web.accounts.context.roles",
                "sonic.web.alerts.context.counts",
            ],
        },
    },
]

# SQLite keeps setup simple for the competition; the schema is plain Django
# models, so PostgreSQL only needs a different DATABASES entry.
DATABASE_DIR = Path(setting("SONIC_DATABASE_DIR", str(BASE_DIR / "database")))
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DATABASE_DIR / "sonic.sqlite3",
    }
}

AUTH_USER_MODEL = "accounts.User"
LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "home"
LOGOUT_REDIRECT_URL = "accounts:login"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = setting("SONIC_TIME_ZONE", "Asia/Karachi")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
# The exported GTM model is loaded by the browser, so it is served as static files at /static/gtm_model/.
STATICFILES_DIRS = [BASE_DIR / "static", ("gtm_model", BASE_DIR / "gtm_model")]
STATIC_ROOT = BASE_DIR / "staticfiles"

# Uploaded audio is never served as a public static file: it lives outside
# the web root and is streamed by a view that checks permissions.
UPLOAD_DIR = Path(setting("SONIC_UPLOAD_DIR", str(AUDIO_DATA_DIR / "web_uploads")))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = False  # the page scripts read it to send the CSRF header
SESSION_COOKIE_AGE = 60 * 60 * 8  # one working shift
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
