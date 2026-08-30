"""Django settings for the Aegis-Credit dashboard."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from cryptography.fernet import Fernet
from django.core.exceptions import ImproperlyConfigured


BASE_DIR = Path(__file__).resolve().parent.parent


TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def env_bool(name: str, default: bool = False) -> bool:
    """Read an explicit boolean and reject typos in security-sensitive flags."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    value = raw_value.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    allowed = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise ImproperlyConfigured(f"{name} must be one of: {allowed}.")


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read a bounded integer without exposing its supplied value in errors."""
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError as exc:
        raise ImproperlyConfigured(f"{name} must be an integer.") from exc
    if minimum is not None and value < minimum:
        raise ImproperlyConfigured(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ImproperlyConfigured(f"{name} must be at most {maximum}.")
    return value


def env_list(name: str, default: str = "") -> list[str]:
    return [value.strip() for value in os.getenv(name, default).split(",") if value.strip()]


def configure_database_tls(database_config: dict[str, object], required: bool) -> None:
    """Require encryption while preserving stronger URL-provided verification modes."""
    if not required:
        return
    options = database_config.setdefault("OPTIONS", {})
    if not isinstance(options, dict):
        raise ImproperlyConfigured("The database OPTIONS setting must be a mapping.")
    ssl_mode = str(options.get("sslmode", "")).strip().lower()
    if not ssl_mode:
        options["sslmode"] = "require"
    elif ssl_mode not in {"require", "verify-ca", "verify-full"}:
        raise ImproperlyConfigured(
            "DATABASE_URL sslmode must be require, verify-ca, or verify-full "
            "when DB_SSL_REQUIRE=True."
        )


def env_secret_mapping(name: str) -> dict[str, str]:
    """Parse a JSON client-to-secret object without silently accepting duplicates."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return {}

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ImproperlyConfigured(f"{name} contains duplicate client id {key!r}.")
            output[key] = value
        return output

    try:
        parsed = json.loads(raw_value, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ImproperlyConfigured(f"{name} must be a valid JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ImproperlyConfigured(f"{name} must be a JSON object mapping client ids to secrets.")

    output: dict[str, str] = {}
    for client_id, secret in parsed.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", client_id):
            raise ImproperlyConfigured(
                f"{name} client ids must be 1-64 letters, numbers, dots, underscores, or hyphens."
            )
        if not isinstance(secret, str) or secret != secret.strip() or len(secret) < 32:
            raise ImproperlyConfigured(
                f"{name} secret for client {client_id!r} must contain at least 32 characters "
                "and no surrounding whitespace."
            )
        output[client_id] = secret
    if len(set(output.values())) != len(output):
        raise ImproperlyConfigured(f"{name} must use a distinct secret for each client id.")
    return output


def merge_legacy_scoring_key(
    configured_keys: dict[str, str],
    legacy_key: str,
) -> dict[str, str]:
    """Add the compatibility credential without weakening client attribution."""
    output = dict(configured_keys)
    if not legacy_key:
        return output
    configured_legacy_key = output.get("legacy")
    if configured_legacy_key and configured_legacy_key != legacy_key:
        raise ImproperlyConfigured(
            "SCORING_API_KEY conflicts with the 'legacy' entry in SCORING_API_KEYS."
        )
    if any(
        client_id != "legacy" and secret == legacy_key
        for client_id, secret in output.items()
    ):
        raise ImproperlyConfigured(
            "SCORING_API_KEY must not reuse another client's SCORING_API_KEYS secret."
        )
    output.setdefault("legacy", legacy_key)
    return output


DEBUG = env_bool("DEBUG", False)


def required_env(name: str, *, minimum_length: int | None = None) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ImproperlyConfigured(f"{name} must be supplied by the environment or a secret manager.")
    if minimum_length is not None and len(value) < minimum_length:
        raise ImproperlyConfigured(f"{name} must contain at least {minimum_length} characters.")
    return value


SECRET_KEY = required_env("SECRET_KEY", minimum_length=50)
AUDIT_HMAC_KEY = required_env("AUDIT_HMAC_KEY", minimum_length=32)
AUDIT_HMAC_KEYS = env_list("AUDIT_HMAC_KEYS") or [AUDIT_HMAC_KEY]
if AUDIT_HMAC_KEYS[0] != AUDIT_HMAC_KEY:
    raise ImproperlyConfigured(
        "AUDIT_HMAC_KEYS must list AUDIT_HMAC_KEY first as the active write key."
    )
if len(set(AUDIT_HMAC_KEYS)) != len(AUDIT_HMAC_KEYS):
    raise ImproperlyConfigured("AUDIT_HMAC_KEYS must not contain duplicate keys.")
FIELD_ENCRYPTION_KEY = required_env("FIELD_ENCRYPTION_KEY")
FIELD_ENCRYPTION_KEYS = env_list("FIELD_ENCRYPTION_KEYS") or [FIELD_ENCRYPTION_KEY]
if FIELD_ENCRYPTION_KEYS[0] != FIELD_ENCRYPTION_KEY:
    raise ImproperlyConfigured(
        "FIELD_ENCRYPTION_KEYS must list FIELD_ENCRYPTION_KEY first as the active write key."
    )
if len(set(FIELD_ENCRYPTION_KEYS)) != len(FIELD_ENCRYPTION_KEYS):
    raise ImproperlyConfigured("FIELD_ENCRYPTION_KEYS must not contain duplicate keys.")
MODEL_SIGNING_PUBLIC_KEY = required_env("MODEL_SIGNING_PUBLIC_KEY")
BACKUP_ENCRYPTION_KEY = required_env("BACKUP_ENCRYPTION_KEY")
try:
    for field_key in FIELD_ENCRYPTION_KEYS:
        Fernet(field_key.encode("ascii"))
    Fernet(BACKUP_ENCRYPTION_KEY.encode("ascii"))
    model_signing_key = base64.b64decode(MODEL_SIGNING_PUBLIC_KEY, validate=True)
except (ValueError, UnicodeEncodeError) as exc:
    raise ImproperlyConfigured(
        "FIELD_ENCRYPTION_KEY, every FIELD_ENCRYPTION_KEYS entry, and "
        "BACKUP_ENCRYPTION_KEY must be valid Fernet keys, "
        "and MODEL_SIGNING_PUBLIC_KEY must be valid base64."
    ) from exc
if len(model_signing_key) != 32:
    raise ImproperlyConfigured("MODEL_SIGNING_PUBLIC_KEY must contain a 32-byte Ed25519 public key.")

ALLOWED_HOSTS = env_list(
    "ALLOWED_HOSTS",
    "127.0.0.1,localhost,testserver",
)
render_hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if render_hostname:
    ALLOWED_HOSTS.append(render_hostname)

CSRF_TRUSTED_ORIGINS = env_list("CSRF_TRUSTED_ORIGINS")
if render_hostname:
    CSRF_TRUSTED_ORIGINS.append(f"https://{render_hostname}")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "app.apps.RiskDashboardConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "app.middleware.ResponseProtectionMiddleware",
]
if not DEBUG:
    MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")

ROOT_URLCONF = "aegis_credit.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.template.context_processors.static",
                "django.contrib.auth.context_processors.auth",
                "app.context_processors.product_shell",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "aegis_credit.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {"timeout": 20},
    }
}
database_url = os.getenv("DATABASE_URL")
DB_SSL_REQUIRE = env_bool("DB_SSL_REQUIRE", not DEBUG)
if database_url:
    import dj_database_url

    DATABASES["default"] = dj_database_url.parse(
        database_url,
        conn_max_age=env_int("DB_CONN_MAX_AGE", 600, minimum=0),
        conn_health_checks=True,
        ssl_require=False,
    )
    configure_database_tls(DATABASES["default"], DB_SSL_REQUIRE)

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = os.getenv("MEDIA_URL", "/media/").strip()
if not MEDIA_URL:
    raise ImproperlyConfigured("MEDIA_URL must not be blank.")
media_root_value = os.getenv("MEDIA_ROOT", str(BASE_DIR / "media")).strip()
if not media_root_value:
    raise ImproperlyConfigured("MEDIA_ROOT must not be blank.")
MEDIA_ROOT = Path(media_root_value)
if not MEDIA_ROOT.is_absolute():
    MEDIA_ROOT = BASE_DIR / MEDIA_ROOT
MEDIA_STORAGE_BACKEND = os.getenv(
    "MEDIA_STORAGE_BACKEND",
    "django.core.files.storage.FileSystemStorage",
).strip()
if not MEDIA_STORAGE_BACKEND:
    raise ImproperlyConfigured("MEDIA_STORAGE_BACKEND must not be blank.")

STORAGES = {
    "default": {
        "BACKEND": MEDIA_STORAGE_BACKEND,
    },
    "staticfiles": {
        "BACKEND": (
            "django.contrib.staticfiles.storage.StaticFilesStorage"
            if DEBUG
            else "whitenoise.storage.CompressedManifestStaticFilesStorage"
        ),
    },
}

TRUST_X_FORWARDED_PROTO = env_bool("TRUST_X_FORWARDED_PROTO", bool(render_hostname))
SECURE_PROXY_SSL_HEADER = (
    ("HTTP_X_FORWARDED_PROTO", "https") if TRUST_X_FORWARDED_PROTO else None
)
SECURE_SSL_REDIRECT = env_bool("SECURE_SSL_REDIRECT", not DEBUG)
SECURE_HSTS_SECONDS = env_int(
    "SECURE_HSTS_SECONDS",
    31536000 if not DEBUG else 0,
    minimum=0,
)
SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool("SECURE_HSTS_INCLUDE_SUBDOMAINS", not DEBUG)
SECURE_HSTS_PRELOAD = env_bool("SECURE_HSTS_PRELOAD", not DEBUG)
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
X_FRAME_OPTIONS = "DENY"
SCORING_API_KEY = os.getenv("SCORING_API_KEY", "")
if SCORING_API_KEY != SCORING_API_KEY.strip():
    raise ImproperlyConfigured("SCORING_API_KEY must not contain surrounding whitespace.")
SCORING_API_KEYS = merge_legacy_scoring_key(
    env_secret_mapping("SCORING_API_KEYS"),
    SCORING_API_KEY,
)
LOGIN_REQUIRED = env_bool("LOGIN_REQUIRED", True)
DATA_PROVENANCE_VERIFIED = env_bool("DATA_PROVENANCE_VERIFIED", False)
LOCAL_DEMO_MODE = env_bool("LOCAL_DEMO_MODE", False)
DEPLOYMENT_TENANT_ID = os.getenv("DEPLOYMENT_TENANT_ID", "").strip()
CASE_RETENTION_DAYS = env_int("CASE_RETENTION_DAYS", 365, minimum=1)
ACCESS_LOG_RETENTION_DAYS = env_int(
    "ACCESS_LOG_RETENTION_DAYS",
    CASE_RETENTION_DAYS,
    minimum=1,
)
MAX_BATCH_ROWS = env_int("MAX_BATCH_ROWS", 1000, minimum=1)
MAX_UPLOAD_BYTES = env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024, minimum=1)
MAX_XLSX_UNCOMPRESSED_BYTES = env_int(
    "MAX_XLSX_UNCOMPRESSED_BYTES",
    50 * 1024 * 1024,
    minimum=1,
)
MAX_XLSX_ARCHIVE_MEMBERS = env_int("MAX_XLSX_ARCHIVE_MEMBERS", 2000, minimum=1)
API_RATE_LIMIT_PER_MINUTE = env_int("API_RATE_LIMIT_PER_MINUTE", 60, minimum=1)
CASE_REVIEW_SLA_HOURS = env_int("CASE_REVIEW_SLA_HOURS", 48, minimum=1)
CASE_PAGE_SIZE = env_int("CASE_PAGE_SIZE", 50, minimum=1)
MONITORING_FRESHNESS_HOURS = env_int("MONITORING_FRESHNESS_HOURS", 24, minimum=1)
MONITORING_MIN_SAMPLE_SIZE = env_int("MONITORING_MIN_SAMPLE_SIZE", 100, minimum=2)
CURRENCY_CODE = os.getenv("CURRENCY_CODE", "").strip().upper()
if CURRENCY_CODE and not re.fullmatch(r"[A-Z]{3}", CURRENCY_CODE):
    raise ImproperlyConfigured("CURRENCY_CODE must be a three-letter ISO 4217 code.")
BATCH_PROCESS_INLINE = env_bool("BATCH_PROCESS_INLINE", DEBUG)
BATCH_LEASE_SECONDS = env_int(
    "BATCH_LEASE_SECONDS",
    300,
    minimum=30,
    maximum=86400,
)
BATCH_MAX_ATTEMPTS = env_int("BATCH_MAX_ATTEMPTS", 3, minimum=1, maximum=20)


def validate_runtime_configuration(
    *,
    debug: bool,
    login_required: bool,
    local_demo_mode: bool,
    data_provenance_verified: bool,
    configured_database_url: str | None,
    database_engine: str,
    database_ssl_required: bool,
    secure_ssl_redirect: bool,
    scoring_api_key: str,
    deployment_tenant_id: str = "",
) -> None:
    if scoring_api_key and len(scoring_api_key) < 32:
        raise ImproperlyConfigured("SCORING_API_KEY must contain at least 32 characters when enabled.")
    if local_demo_mode and not debug:
        raise ImproperlyConfigured("LOCAL_DEMO_MODE is allowed only when DEBUG=True.")
    if local_demo_mode and data_provenance_verified:
        raise ImproperlyConfigured(
            "LOCAL_DEMO_MODE and DATA_PROVENANCE_VERIFIED cannot both be enabled."
        )
    if not debug and not login_required:
        raise ImproperlyConfigured("LOGIN_REQUIRED cannot be disabled when DEBUG=False.")
    if not debug and not configured_database_url:
        raise ImproperlyConfigured("DATABASE_URL is required when DEBUG=False; SQLite is local-only.")
    if not debug and "postgresql" not in database_engine:
        raise ImproperlyConfigured("PostgreSQL is required when DEBUG=False.")
    if not debug and not database_ssl_required:
        raise ImproperlyConfigured("DB_SSL_REQUIRE cannot be disabled when DEBUG=False.")
    if not debug and not secure_ssl_redirect:
        raise ImproperlyConfigured("SECURE_SSL_REDIRECT cannot be disabled when DEBUG=False.")
    if not debug and not deployment_tenant_id:
        raise ImproperlyConfigured(
            "DEPLOYMENT_TENANT_ID is required in shared deployments; "
            "Aegis-Credit supports one tenant per deployment."
        )


validate_runtime_configuration(
    debug=DEBUG,
    login_required=LOGIN_REQUIRED,
    local_demo_mode=LOCAL_DEMO_MODE,
    data_provenance_verified=DATA_PROVENANCE_VERIFIED,
    configured_database_url=database_url,
    database_engine=str(DATABASES["default"]["ENGINE"]),
    database_ssl_required=DB_SSL_REQUIRE,
    secure_ssl_redirect=SECURE_SSL_REDIRECT,
    scoring_api_key=SCORING_API_KEY,
    deployment_tenant_id=DEPLOYMENT_TENANT_ID,
)

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "overview"
LOGOUT_REDIRECT_URL = "login"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

CACHES = {
    "default": {
        "BACKEND": os.getenv(
            "CACHE_BACKEND",
            "django.core.cache.backends.locmem.LocMemCache",
        ),
        "LOCATION": os.getenv("CACHE_LOCATION", "aegis-credit"),
    }
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "{asctime} {levelname} {name}: {message}",
            "style": "{",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        }
    },
    "root": {
        "handlers": ["console"],
        "level": os.getenv("LOG_LEVEL", "INFO"),
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
