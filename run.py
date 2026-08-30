#!/usr/bin/env python3
"""Start Aegis-Credit locally with one command.

Run ``python run.py`` (or ``py run.py`` on Windows).  The script keeps all
local-only configuration in ``.aegis-credit-local.env`` so encryption keys stay
stable between launches and existing local cases remain readable.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import secrets
import stat
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements-app.txt"
LOCAL_ENV = PROJECT_ROOT / ".aegis-credit-local.env"
DOCKER_ENV = PROJECT_ROOT / ".aegis-credit-docker.env"
REQUIREMENTS_MARKER = VENV_DIR / ".aegis-credit-requirements.sha256"


def venv_python() -> Path:
    """Return the virtual-environment interpreter for the active platform."""
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def require_supported_python() -> None:
    if sys.version_info < (3, 12):
        raise SystemExit(
            "Aegis-Credit requires Python 3.12 or newer. "
            "Install a supported Python version and run this command again."
        )


def create_venv_if_needed() -> Path:
    python = venv_python()
    if python.exists():
        return python

    print("Creating local Python environment...")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    if not python.exists():
        raise RuntimeError("The local Python environment was not created successfully.")
    return python


def requirements_digest() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def install_dependencies(python: Path) -> None:
    digest = requirements_digest()
    installed_digest = REQUIREMENTS_MARKER.read_text().strip() if REQUIREMENTS_MARKER.exists() else ""
    if digest == installed_digest:
        return

    print("Installing required packages (this may take a few minutes the first time)...")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)],
        check=True,
    )
    REQUIREMENTS_MARKER.write_text(f"{digest}\n", encoding="utf-8")


def verify_cryptography_backend(python: Path) -> None:
    """Fail early with an actionable message when Windows blocks CFFI.

    ``cryptography`` loads CFFI while Django imports the settings module.  On
    managed Windows devices, application-control policies can deny that native
    extension before Django has a chance to show a useful error.  Do a tiny
    preflight import so the launcher can identify that situation precisely.
    """
    result = subprocess.run(
        [str(python), "-c", "from cryptography.fernet import Fernet"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    error = f"{result.stdout}\n{result.stderr}".lower()
    if os.name == "nt" and "_cffi_backend" in error and "dll load failed" in error:
        raise SystemExit(
            "\nWindows application control blocked CFFI, a required native "
            "component of cryptography. Aegis-Credit cannot safely run without "
            "its encryption library (and the dashboard also needs native NumPy "
            "and scikit-learn modules). Ask your IT administrator to allow native "
            "*.pyd extensions in this project's .venv, including "
            "_cffi_backend*.pyd, then run python run.py again. If your "
            "organisation provides an approved Docker Desktop installation, you "
            "can instead start Docker Desktop and use Start Aegis-Credit "
            "Docker.bat.\n"
        )

    details = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
    raise SystemExit(f"Unable to load the required cryptography package:\n{details}")


def parse_env_file(path: Path) -> dict[str, str]:
    """Read the deliberately simple KEY=VALUE local launcher file."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip()
    return values


def fernet_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def create_local_environment() -> dict[str, str]:
    """Create stable, untracked development settings without touching ``.env``."""
    values = parse_env_file(LOCAL_ENV)
    defaults = {
        "DEBUG": "True",
        "SECURE_SSL_REDIRECT": "False",
        "ALLOWED_HOSTS": "127.0.0.1,localhost",
        "LOGIN_REQUIRED": "False",
        # Local demo mode loads the bundled model after checking it against the
        # checked-in SHA-256. Production deployments leave this disabled and
        # require a signed, provenance-approved release.
        "LOCAL_DEMO_MODE": "True",
        # The included demonstration data is deliberately not an operational
        # model release. Keep the governance gate in place for local demos too.
        "DATA_PROVENANCE_VERIFIED": "False",
        "SECRET_KEY": secrets.token_urlsafe(48),
        "AUDIT_HMAC_KEY": secrets.token_urlsafe(48),
        "AUDIT_HMAC_KEYS": "",
        "FIELD_ENCRYPTION_KEY": fernet_key(),
        "BACKUP_ENCRYPTION_KEY": fernet_key(),
        "MODEL_SIGNING_PUBLIC_KEY": base64.b64encode(os.urandom(32)).decode("ascii"),
    }
    changed = False
    for name, value in defaults.items():
        if not values.get(name):
            values[name] = value
            changed = True

    if changed or not LOCAL_ENV.exists():
        header = (
            "# Generated for local development by run.py. Keep this file private: it\n"
            "# contains keys needed to read locally stored encrypted case data.\n"
        )
        contents = header + "".join(f"{name}={value}\n" for name, value in sorted(values.items()))
        LOCAL_ENV.write_text(contents, encoding="utf-8")
        print("Created local development settings in .aegis-credit-local.env.")
    if os.name != "nt":
        LOCAL_ENV.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return values


def create_local_docker_environment() -> dict[str, str]:
    """Create stable, local-only Compose settings without weakening production manifests."""
    values = parse_env_file(DOCKER_ENV)
    defaults = {
        "DEBUG": "True",
        "SECURE_SSL_REDIRECT": "False",
        "TRUST_X_FORWARDED_PROTO": "False",
        "ALLOWED_HOSTS": "127.0.0.1,localhost",
        "LOGIN_REQUIRED": "False",
        "LOCAL_DEMO_MODE": "True",
        "DATA_PROVENANCE_VERIFIED": "False",
        "SECRET_KEY": secrets.token_urlsafe(48),
        "AUDIT_HMAC_KEY": secrets.token_urlsafe(48),
        "AUDIT_HMAC_KEYS": "",
        "FIELD_ENCRYPTION_KEY": fernet_key(),
        "BACKUP_ENCRYPTION_KEY": fernet_key(),
        "MODEL_SIGNING_PUBLIC_KEY": base64.b64encode(os.urandom(32)).decode("ascii"),
        "POSTGRES_DB": "aegis_credit",
        "POSTGRES_USER": "aegis_credit",
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "DB_SSL_REQUIRE": "False",
        "SCORING_API_KEY": "",
        "SCORING_API_KEYS": "{}",
        "API_RATE_LIMIT_PER_MINUTE": "60",
        "CASE_RETENTION_DAYS": "365",
        "ACCESS_LOG_RETENTION_DAYS": "365",
        "RETENTION_INTERVAL_SECONDS": "86400",
        "MAX_BATCH_ROWS": "1000",
        "MAX_UPLOAD_BYTES": str(10 * 1024 * 1024),
        "MAX_XLSX_UNCOMPRESSED_BYTES": str(50 * 1024 * 1024),
        "MAX_XLSX_ARCHIVE_MEMBERS": "2000",
        "CASE_REVIEW_SLA_HOURS": "48",
        "CASE_PAGE_SIZE": "50",
        "MONITORING_FRESHNESS_HOURS": "24",
        "MONITORING_MIN_SAMPLE_SIZE": "100",
        "CURRENCY_CODE": "",
        "BATCH_PROCESS_INLINE": "False",
        "BATCH_LEASE_SECONDS": "300",
        "BATCH_MAX_ATTEMPTS": "3",
        "MEDIA_ROOT": "/app/media",
        "MEDIA_URL": "/media/",
        "MEDIA_STORAGE_BACKEND": "django.core.files.storage.FileSystemStorage",
        "BOOTSTRAP_ADMIN_USERNAME": "",
        "BOOTSTRAP_ADMIN_PASSWORD": "",
        "BOOTSTRAP_ADMIN_EMAIL": "",
    }
    changed = False
    values_allowed_to_be_blank = {
        "SCORING_API_KEY",
        "CURRENCY_CODE",
        "BOOTSTRAP_ADMIN_USERNAME",
        "BOOTSTRAP_ADMIN_PASSWORD",
        "BOOTSTRAP_ADMIN_EMAIL",
    }
    for name, value in defaults.items():
        if name not in values or (name not in values_allowed_to_be_blank and not values[name]):
            values[name] = value
            changed = True

    if not values.get("DATABASE_URL"):
        values["DATABASE_URL"] = (
            f"postgresql://{values['POSTGRES_USER']}:{values['POSTGRES_PASSWORD']}"
            f"@database:5432/{values['POSTGRES_DB']}"
        )
        changed = True

    if changed or not DOCKER_ENV.exists():
        header = (
            "# Generated by run.py for LOCAL DOCKER DEMONSTRATION USE ONLY.\n"
            "# Never deploy, commit, publish, or share this file or its secrets.\n"
        )
        contents = header + "".join(f"{name}={value}\n" for name, value in sorted(values.items()))
        DOCKER_ENV.write_text(contents, encoding="utf-8")
    if os.name != "nt":
        DOCKER_ENV.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"Local Docker settings are ready in {DOCKER_ENV.name}.")
    return values


def command_environment() -> dict[str, str]:
    # This is a local-only launcher. Its saved configuration deliberately wins
    # over ambient CI/editor variables (for example DEBUG=release), which could
    # otherwise make a development server look for a production static manifest.
    # Edit .aegis-credit-local.env when an intentional local override is needed.
    environment = os.environ.copy()
    environment.update(create_local_environment())
    return environment


def run_manage(python: Path, environment: dict[str, str], *arguments: str) -> None:
    subprocess.run([str(python), "manage.py", *arguments], cwd=PROJECT_ROOT, env=environment, check=True)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aegis-Credit locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the local URL in a browser.")
    parser.add_argument("--check", action="store_true", help="Prepare the project and run Django checks without starting the server.")
    parser.add_argument(
        "launcher_command",
        nargs="?",
        choices=("manage", "docker-env"),
        help="Run a local management command or prepare local-only Docker settings.",
    )
    parser.add_argument("manage_arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(arguments)
    if parsed.launcher_command == "manage" and not parsed.manage_arguments:
        parser.error("manage requires a Django management command, for example: run.py manage check")
    if parsed.launcher_command == "docker-env" and parsed.manage_arguments:
        parser.error("docker-env does not accept additional arguments")
    return parsed


def main() -> None:
    args = parse_args()
    require_supported_python()
    if args.launcher_command == "docker-env":
        create_local_docker_environment()
        print(
            "Start the local stack with: docker compose --env-file "
            f"{DOCKER_ENV.name} up --build"
        )
        return
    python = create_venv_if_needed()
    install_dependencies(python)
    verify_cryptography_backend(python)
    environment = command_environment()

    if args.launcher_command == "manage":
        run_manage(python, environment, *args.manage_arguments)
        return

    print("Preparing the local database...")
    run_manage(python, environment, "migrate", "--no-input")
    run_manage(python, environment, "bootstrap_roles")
    if args.check:
        run_manage(python, environment, "check")
        print("Aegis-Credit is ready to start with: python run.py")
        return

    url = f"http://{args.host}:{args.port}/"
    print(f"\nAegis-Credit is running at {url}")
    print("Press Ctrl+C to stop it.\n")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    try:
        run_manage(python, environment, "runserver", f"{args.host}:{args.port}")
    except KeyboardInterrupt:
        print("\nAegis-Credit stopped.")


if __name__ == "__main__":
    main()
