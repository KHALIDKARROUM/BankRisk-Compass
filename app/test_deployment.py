from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.core.exceptions import ImproperlyConfigured
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase

import run as launcher
from app.backup import (
    BackupIntegrityError,
    PostgresConnection,
    backup_key,
    decrypt_stream,
    encrypt_stream,
    postgres_connection,
)
from aegis_credit import settings as project_settings


class EnvironmentSettingTests(SimpleTestCase):
    def test_boolean_parser_uses_default_for_missing_or_blank_values(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AEGIS_CREDIT_TEST_BOOLEAN", None)
            self.assertTrue(project_settings.env_bool("AEGIS_CREDIT_TEST_BOOLEAN", True))
        with patch.dict(os.environ, {"AEGIS_CREDIT_TEST_BOOLEAN": "   "}, clear=False):
            self.assertFalse(project_settings.env_bool("AEGIS_CREDIT_TEST_BOOLEAN", False))

    def test_boolean_parser_rejects_a_typo(self) -> None:
        with patch.dict(os.environ, {"AEGIS_CREDIT_TEST_BOOLEAN": "treu"}, clear=False):
            with self.assertRaises(ImproperlyConfigured):
                project_settings.env_bool("AEGIS_CREDIT_TEST_BOOLEAN")

    def test_integer_parser_enforces_its_lower_bound(self) -> None:
        with patch.dict(os.environ, {"AEGIS_CREDIT_TEST_INTEGER": "0"}, clear=False):
            with self.assertRaises(ImproperlyConfigured):
                project_settings.env_int("AEGIS_CREDIT_TEST_INTEGER", 10, minimum=1)

    def test_database_tls_preserves_verification_and_rejects_downgrades(self) -> None:
        database = {"OPTIONS": {"sslmode": "verify-full"}}
        project_settings.configure_database_tls(database, True)
        self.assertEqual(database["OPTIONS"]["sslmode"], "verify-full")

        database_without_mode: dict[str, object] = {}
        project_settings.configure_database_tls(database_without_mode, True)
        self.assertEqual(database_without_mode["OPTIONS"]["sslmode"], "require")

        with self.assertRaisesRegex(ImproperlyConfigured, "sslmode"):
            project_settings.configure_database_tls(
                {"OPTIONS": {"sslmode": "disable"}},
                True,
            )

    def test_scoring_api_key_mapping_accepts_independent_clients(self) -> None:
        first_secret = "a" * 32
        second_secret = "b" * 48
        with patch.dict(
            os.environ,
            {
                "AEGIS_CREDIT_TEST_API_KEYS": (
                    '{"underwriter-app":"' + first_secret + '",'
                    '"batch.v2":"' + second_secret + '"}'
                )
            },
            clear=False,
        ):
            self.assertEqual(
                project_settings.env_secret_mapping("AEGIS_CREDIT_TEST_API_KEYS"),
                {"underwriter-app": first_secret, "batch.v2": second_secret},
            )

    def test_scoring_api_key_mapping_rejects_ambiguous_credentials(self) -> None:
        invalid_values = (
            "[]",
            '{"client":"too-short"}',
            '{"client":"' + ("a" * 32) + '","client":"' + ("b" * 32) + '"}',
            '{"client-a":"' + ("a" * 32) + '","client-b":"' + ("a" * 32) + '"}',
        )
        for value in invalid_values:
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"AEGIS_CREDIT_TEST_API_KEYS": value},
                clear=False,
            ):
                with self.assertRaises(ImproperlyConfigured):
                    project_settings.env_secret_mapping("AEGIS_CREDIT_TEST_API_KEYS")

    def test_legacy_scoring_key_is_exposed_as_an_attributed_client(self) -> None:
        legacy_secret = "l" * 32
        self.assertEqual(
            project_settings.merge_legacy_scoring_key({}, legacy_secret),
            {"legacy": legacy_secret},
        )
        with self.assertRaisesRegex(ImproperlyConfigured, "must not reuse"):
            project_settings.merge_legacy_scoring_key(
                {"partner": legacy_secret},
                legacy_secret,
            )

    def test_production_cannot_disable_login_or_use_sqlite(self) -> None:
        with self.assertRaisesRegex(ImproperlyConfigured, "LOGIN_REQUIRED"):
            project_settings.validate_runtime_configuration(
                debug=False,
                login_required=False,
                local_demo_mode=False,
                data_provenance_verified=False,
                configured_database_url="postgresql://database/example",
                database_engine="django.db.backends.postgresql",
                database_ssl_required=True,
                secure_ssl_redirect=True,
                scoring_api_key="",
                deployment_tenant_id="client-a",
            )
        with self.assertRaisesRegex(ImproperlyConfigured, "DATABASE_URL"):
            project_settings.validate_runtime_configuration(
                debug=False,
                login_required=True,
                local_demo_mode=False,
                data_provenance_verified=False,
                configured_database_url=None,
                database_engine="django.db.backends.sqlite3",
                database_ssl_required=True,
                secure_ssl_redirect=True,
                scoring_api_key="",
                deployment_tenant_id="client-a",
            )

    def test_demo_mode_is_rejected_in_production(self) -> None:
        with self.assertRaisesRegex(ImproperlyConfigured, "LOCAL_DEMO_MODE"):
            project_settings.validate_runtime_configuration(
                debug=False,
                login_required=True,
                local_demo_mode=True,
                data_provenance_verified=False,
                configured_database_url="postgresql://database/example",
                database_engine="django.db.backends.postgresql",
                database_ssl_required=True,
                secure_ssl_redirect=True,
                scoring_api_key="",
            )

    def test_production_requires_postgresql_database_tls_and_https(self) -> None:
        safe_values = {
            "debug": False,
            "login_required": True,
            "local_demo_mode": False,
            "data_provenance_verified": False,
            "configured_database_url": "postgresql://database/example",
            "database_engine": "django.db.backends.postgresql",
            "database_ssl_required": True,
            "secure_ssl_redirect": True,
            "scoring_api_key": "",
            "deployment_tenant_id": "client-a",
        }
        for name, value, message in (
            ("database_engine", "django.db.backends.sqlite3", "PostgreSQL"),
            ("database_ssl_required", False, "DB_SSL_REQUIRE"),
            ("secure_ssl_redirect", False, "SECURE_SSL_REDIRECT"),
        ):
            configured_values = {**safe_values, name: value}
            with self.subTest(name=name), self.assertRaisesRegex(
                ImproperlyConfigured,
                message,
            ):
                project_settings.validate_runtime_configuration(**configured_values)

    def test_shared_deployment_requires_a_tenant_identifier(self) -> None:
        with self.assertRaisesRegex(ImproperlyConfigured, "DEPLOYMENT_TENANT_ID"):
            project_settings.validate_runtime_configuration(
                debug=False,
                login_required=True,
                local_demo_mode=False,
                data_provenance_verified=False,
                configured_database_url="postgresql://database/example",
                database_engine="django.db.backends.postgresql",
                database_ssl_required=True,
                secure_ssl_redirect=True,
                scoring_api_key="",
                deployment_tenant_id="",
            )

    def test_new_operational_limits_have_safe_defaults(self) -> None:
        self.assertEqual(project_settings.MAX_XLSX_UNCOMPRESSED_BYTES, 50 * 1024 * 1024)
        self.assertEqual(project_settings.MAX_XLSX_ARCHIVE_MEMBERS, 2000)
        self.assertEqual(project_settings.CASE_REVIEW_SLA_HOURS, 48)
        self.assertEqual(project_settings.CASE_PAGE_SIZE, 50)
        self.assertEqual(project_settings.MONITORING_FRESHNESS_HOURS, 24)
        self.assertEqual(project_settings.MONITORING_MIN_SAMPLE_SIZE, 100)
        self.assertEqual(project_settings.CURRENCY_CODE, "")
        self.assertEqual(project_settings.BATCH_PROCESS_INLINE, project_settings.DEBUG)
        self.assertEqual(project_settings.BATCH_LEASE_SECONDS, 300)
        self.assertEqual(project_settings.BATCH_MAX_ATTEMPTS, 3)
        self.assertIn("default", project_settings.STORAGES)


class LocalLauncherTests(SimpleTestCase):
    def test_manage_subcommand_preserves_django_arguments(self) -> None:
        parsed = launcher.parse_args(["manage", "check", "--deploy"])
        self.assertEqual(parsed.launcher_command, "manage")
        self.assertEqual(parsed.manage_arguments, ["check", "--deploy"])

    def test_local_environment_file_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".aegis-credit-local.env"
            with patch.object(launcher, "LOCAL_ENV", path):
                first = launcher.create_local_environment()
                second = launcher.create_local_environment()
            self.assertEqual(first, second)
            self.assertEqual(path.read_text(encoding="utf-8").count("SECRET_KEY="), 1)

    def test_docker_environment_is_local_only_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".aegis-credit-docker.env"
            with patch.object(launcher, "DOCKER_ENV", path):
                first = launcher.create_local_docker_environment()
                second = launcher.create_local_docker_environment()
            contents = path.read_text(encoding="utf-8")
        self.assertEqual(first, second)
        self.assertEqual(first["DEBUG"], "True")
        self.assertEqual(first["LOCAL_DEMO_MODE"], "True")
        self.assertEqual(first["SCORING_API_KEYS"], "{}")
        self.assertEqual(first["BATCH_LEASE_SECONDS"], "300")
        self.assertEqual(first["BATCH_MAX_ATTEMPTS"], "3")
        self.assertEqual(first["MONITORING_MIN_SAMPLE_SIZE"], "100")
        self.assertEqual(first["CURRENCY_CODE"], "")
        self.assertIn(first["POSTGRES_PASSWORD"], first["DATABASE_URL"])
        self.assertIn("LOCAL DOCKER DEMONSTRATION USE ONLY", contents)


class EncryptedBackupTests(SimpleTestCase):
    def test_database_password_is_kept_out_of_process_arguments(self) -> None:
        database = {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": "aegis_credit",
            "USER": "backup_user",
            "PASSWORD": "do-not-put-this-in-argv",
            "HOST": "database",
            "PORT": "5432",
            "OPTIONS": {"sslmode": "verify-full"},
        }
        with patch.dict(project_settings.DATABASES, {"default": database}, clear=True):
            connection = postgres_connection()
        self.assertNotIn(database["PASSWORD"], " ".join(connection.command_arguments))
        self.assertEqual(connection.environment["PGPASSWORD"], database["PASSWORD"])
        self.assertEqual(connection.environment["PGSSLMODE"], "verify-full")
        self.assertNotIn("SECRET_KEY", connection.environment)

    def test_streaming_backup_round_trip_and_tamper_detection(self) -> None:
        key = backup_key()
        plaintext = (b"aegis-credit-test-row\n" * 100_000) + b"done"
        encrypted = io.BytesIO()
        encrypt_stream(io.BytesIO(plaintext), encrypted, key)

        restored = io.BytesIO()
        encrypted.seek(0)
        decrypt_stream(encrypted, restored, key)
        self.assertEqual(restored.getvalue(), plaintext)

        tampered = bytearray(encrypted.getvalue())
        tampered[len(tampered) // 2] ^= 1
        with self.assertRaises(BackupIntegrityError):
            decrypt_stream(io.BytesIO(tampered), None, key)

    def test_restore_command_defaults_to_authenticated_dry_run(self) -> None:
        key = backup_key()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backup.dump.brc"
            with path.open("wb") as destination:
                encrypt_stream(io.BytesIO(b"test pg dump"), destination, key)
            output = io.StringIO()
            call_command("restore_database", backup=str(path), stdout=output)
        self.assertIn("Dry run only", output.getvalue())

    def test_backup_command_streams_to_an_atomic_encrypted_file(self) -> None:
        key = backup_key()
        process = MagicMock()
        process.stdout = io.BytesIO(b"postgres-custom-format-dump")
        process.wait.return_value = 0
        connection = PostgresConnection("aegis_credit", (), {"PATH": os.environ.get("PATH", "")})
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.management.commands.backup_database.backup_key",
            return_value=key,
        ), patch(
            "app.management.commands.backup_database.postgres_connection",
            return_value=connection,
        ), patch(
            "app.management.commands.backup_database.subprocess.Popen",
            return_value=process,
        ):
            call_command("backup_database", destination=directory, stdout=io.StringIO())
            backup_files = list(Path(directory).glob("*.dump.brc"))
            temporary_files = list(Path(directory).glob("*.tmp"))
            self.assertEqual(len(backup_files), 1)
            restored = io.BytesIO()
            with backup_files[0].open("rb") as source:
                decrypt_stream(source, restored, key)
        self.assertEqual(temporary_files, [])
        self.assertEqual(restored.getvalue(), b"postgres-custom-format-dump")

    def test_restore_requires_exact_database_confirmation_before_subprocess(self) -> None:
        key = backup_key()
        connection = PostgresConnection("expected_database", (), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backup.dump.brc"
            with path.open("wb") as destination:
                encrypt_stream(io.BytesIO(b"test pg dump"), destination, key)
            with patch(
                "app.management.commands.restore_database.postgres_connection",
                return_value=connection,
            ), patch("app.management.commands.restore_database.subprocess.Popen") as popen:
                with self.assertRaisesRegex(CommandError, "exactly match"):
                    call_command(
                        "restore_database",
                        backup=str(path),
                        confirm_database="wrong_database",
                    )
                popen.assert_not_called()
