from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import tempfile
import uuid
import zipfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.core.management.base import CommandError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import connection
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from joblib import load as load_joblib

from app import services
from app.batch_processing import process_batch, recover_stale_batches
from app.models import (
    ApiRateLimitBucket,
    AssessmentCase,
    BatchAssessment,
    BatchRow,
    CaseOutcome,
    CaseReviewEvent,
    DataDeletionReceipt,
    LegalHoldEvent,
    MonitoringAcknowledgement,
    MonitoringRun,
    PolicyScenario,
    PolicyScenarioEvent,
    PredictionAudit,
    SensitiveDataAccessLog,
)


def add_to_group(user, name: str) -> None:
    group, _ = Group.objects.get_or_create(name=name)
    user.groups.add(group)


def make_case(
    *,
    created_by=None,
    assigned_to=None,
    batch: BatchAssessment | None = None,
    reference: str = "CASE-001",
    request_id: uuid.UUID | None = None,
    namespace: str = "web:test",
    legal_hold: bool = False,
    probability: float = 0.2,
    model_version: str = "2.2.0",
    release_id: str = "release-test",
) -> AssessmentCase:
    return AssessmentCase.objects.create(
        request_id=request_id or uuid.uuid4(),
        idempotency_namespace=namespace,
        request_digest="request-digest",
        created_by=created_by,
        assigned_to=assigned_to,
        batch=batch,
        source="web",
        applicant_reference=reference,
        applicant_reference_digest=services.reference_digest(reference),
        application_data={
            "person_income": 65_000,
            "person_emp_length": 5,
            "loan_amnt": 8_000,
            "loan_percent_income": 8_000 / 65_000,
            "cb_person_cred_hist_length": 6,
            "person_home_ownership": "RENT",
            "loan_intent": "PERSONAL",
            "cb_person_default_on_file": "N",
        },
        probability=probability,
        threshold=0.3,
        risk_category="Low" if probability < 0.3 else "High",
        screening_result="No elevated repayment concern identified",
        recommendation="Continue with standard review",
        model_version=model_version,
        model_release_id=release_id,
        deployment_stage=AssessmentCase.DeploymentStage.APPROVED,
        explanation_rows=[],
        explanation_method="Test explanation",
        warnings=[],
        legal_hold=legal_hold,
        due_at=timezone.now() + timedelta(hours=48),
    )


def make_audit(case: AssessmentCase, *, namespace: str | None = None) -> PredictionAudit:
    return PredictionAudit.objects.create(
        request_id=case.request_id,
        idempotency_namespace=namespace or case.idempotency_namespace,
        request_digest=case.request_digest,
        feature_digest="f" * 64,
        probability=case.probability,
        threshold=case.threshold,
        risk_category=case.risk_category,
        decision=case.recommendation,
        model_version=case.model_version,
        deployment_stage=case.deployment_stage,
        case=case,
        actor=case.created_by,
    )


@override_settings(LOGIN_REQUIRED=True, LOCAL_DEMO_MODE=False)
class CaseGovernanceWorkflowTests(TestCase):
    def setUp(self) -> None:
        user_model = get_user_model()
        self.analyst = user_model.objects.create_user("case-analyst", password="test-password")
        self.other_analyst = user_model.objects.create_user(
            "other-analyst", password="test-password"
        )
        self.reviewer = user_model.objects.create_user("case-reviewer", password="test-password")
        self.legal = user_model.objects.create_user("legal-officer", password="test-password")
        add_to_group(self.analyst, "Analysts")
        add_to_group(self.other_analyst, "Analysts")
        add_to_group(self.reviewer, "Reviewers")
        add_to_group(self.legal, "Legal Officers")

    def test_analyst_object_scope_hides_other_cases_and_batches(self) -> None:
        own_case = make_case(created_by=self.analyst, reference="OWN-CASE")
        other_case = make_case(created_by=self.other_analyst, reference="OTHER-CASE")
        own_batch = BatchAssessment.objects.create(
            created_by=self.analyst,
            file_name="own.csv",
            upload_payload=b"payload",
        )
        other_batch = BatchAssessment.objects.create(
            created_by=self.other_analyst,
            file_name="other.csv",
            upload_payload=b"payload",
        )
        self.client.force_login(self.analyst)

        self.assertEqual(self.client.get(reverse("case-detail", args=[own_case.id])).status_code, 200)
        self.assertEqual(
            self.client.get(reverse("case-detail", args=[other_case.id])).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(reverse("batch-detail", args=[own_batch.id])).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(reverse("batch-detail", args=[other_batch.id])).status_code,
            404,
        )

    def test_review_transition_creates_append_only_before_and_after_event(self) -> None:
        case = make_case(created_by=self.analyst)
        self.client.force_login(self.reviewer)

        response = self.client.post(
            reverse("case-detail", args=[case.id]),
            {
                "form_action": "review",
                "review-expected_version": 0,
                "review-status": AssessmentCase.Status.IN_REVIEW,
                "review-reviewer_notes": "Affordability evidence checked.",
                "review-override_decision": AssessmentCase.OverrideDecision.MANUAL,
                "review-override_reason": "Income evidence requires a second review.",
            },
        )

        self.assertRedirects(response, reverse("case-detail", args=[case.id]))
        case.refresh_from_db()
        self.assertEqual(case.review_version, 1)
        self.assertEqual(case.status, AssessmentCase.Status.IN_REVIEW)
        self.assertEqual(case.reviewed_by, self.reviewer)
        event = CaseReviewEvent.objects.get(case=case)
        self.assertEqual(event.event_type, CaseReviewEvent.EventType.REVIEW)
        self.assertEqual(event.before_state["status"], AssessmentCase.Status.NEW)
        self.assertEqual(event.after_state["status"], AssessmentCase.Status.IN_REVIEW)
        event.reason = "Attempted rewrite"
        with self.assertRaisesRegex(ValidationError, "immutable"):
            event.save()

    def test_stale_review_version_and_invalid_transition_do_not_mutate_case(self) -> None:
        case = make_case(created_by=self.analyst)
        AssessmentCase.objects.filter(pk=case.pk).update(review_version=2)
        self.client.force_login(self.reviewer)
        base_payload = {
            "form_action": "review",
            "review-reviewer_notes": "Review note",
            "review-override_decision": "",
            "review-override_reason": "",
        }

        stale = self.client.post(
            reverse("case-detail", args=[case.id]),
            {
                **base_payload,
                "review-expected_version": 1,
                "review-status": AssessmentCase.Status.IN_REVIEW,
            },
        )
        self.assertEqual(stale.status_code, 200)
        self.assertContains(stale, "changed in another session")

        invalid = self.client.post(
            reverse("case-detail", args=[case.id]),
            {
                **base_payload,
                "review-expected_version": 2,
                "review-status": AssessmentCase.Status.CLEARED,
            },
        )
        self.assertEqual(invalid.status_code, 200)
        self.assertContains(invalid, "status transition is not allowed")
        case.refresh_from_db()
        self.assertEqual(case.status, AssessmentCase.Status.NEW)
        self.assertEqual(case.review_version, 2)
        self.assertFalse(CaseReviewEvent.objects.filter(case=case).exists())

    def test_only_legal_role_can_place_and_release_a_documented_hold(self) -> None:
        case = make_case(created_by=self.analyst)
        url = reverse("case-detail", args=[case.id])
        payload = {
            "form_action": "legal_hold",
            "legal-action": LegalHoldEvent.Action.PLACED,
            "legal-reason": "Preserve records for an active investigation.",
            "legal-ticket_reference": "LEGAL-2026-001",
            "legal-expected_version": 0,
        }

        self.client.force_login(self.analyst)
        self.assertEqual(self.client.post(url, payload).status_code, 403)
        case.refresh_from_db()
        self.assertFalse(case.legal_hold)

        self.client.force_login(self.legal)
        self.assertRedirects(self.client.post(url, payload), url)
        case.refresh_from_db()
        self.assertTrue(case.legal_hold)
        self.assertEqual(case.review_version, 1)
        placed = LegalHoldEvent.objects.get(case=case)
        self.assertEqual(placed.actor, self.legal)
        self.assertEqual(placed.ticket_reference, "LEGAL-2026-001")

        release_payload = {
            "form_action": "legal_hold",
            "legal-action": LegalHoldEvent.Action.RELEASED,
            "legal-reason": "The preservation obligation has formally ended.",
            "legal-ticket_reference": "LEGAL-2026-001-CLOSE",
            "legal-expected_version": 1,
        }
        self.assertRedirects(self.client.post(url, release_payload), url)
        case.refresh_from_db()
        self.assertFalse(case.legal_hold)
        self.assertEqual(case.review_version, 2)
        self.assertEqual(case.legal_hold_events.count(), 2)

    def test_reviewer_records_one_mature_outcome_and_an_immutable_event(self) -> None:
        case = make_case(created_by=self.analyst)
        self.client.force_login(self.reviewer)
        response = self.client.post(
            reverse("case-detail", args=[case.id]),
            {
                "form_action": "outcome",
                "outcome-outcome": CaseOutcome.Outcome.PERFORMING,
                "outcome-outcome_date": "2025-01-01",
                "outcome-performance_window_end": "2025-12-31",
                "outcome-as_of_date": "2026-01-01",
                "outcome-exposure_at_default": "10000.00",
                "outcome-loss_amount": "0.00",
                "outcome-source": "Servicing system",
                "outcome-source_reference": "PERF-001",
                "outcome-notes": "Twelve-month performance window completed.",
            },
        )

        self.assertRedirects(response, reverse("case-detail", args=[case.id]))
        outcome = CaseOutcome.objects.get(case=case)
        self.assertEqual(outcome.recorded_by, self.reviewer)
        event = CaseReviewEvent.objects.get(case=case)
        self.assertEqual(event.event_type, CaseReviewEvent.EventType.OUTCOME)
        outcome.notes = "Attempted change"
        with self.assertRaisesRegex(ValidationError, "immutable"):
            outcome.save()

    def test_legal_monitoring_access_is_read_only(self) -> None:
        run = MonitoringRun.objects.create(
            as_of_date=date(2026, 8, 1),
            window_start=date(2026, 7, 1),
            window_end=date(2026, 7, 31),
            model_version="2.2.0",
            model_release_id="release-test",
            sample_size=100,
            status=MonitoringRun.Status.ALERT,
            metrics={},
            alerts=[{"feature": "person_income", "status": "alert"}],
            input_digest="d" * 64,
            owner="Model Risk",
        )
        self.client.force_login(self.legal)
        self.assertEqual(self.client.get(reverse("monitoring")).status_code, 200)
        response = self.client.post(
            reverse("monitoring-acknowledge", args=[run.id]),
            {"action": "acknowledged", "note": "Legal review acknowledgement."},
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(MonitoringAcknowledgement.objects.filter(run=run).exists())


class ScoringApiWorkflowTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.bundle = load_joblib(services.MODEL_PATH)

    def setUp(self) -> None:
        self.model_loader = patch("app.services.load_model_bundle", return_value=self.bundle)
        self.model_loader.start()
        self.addCleanup(self.model_loader.stop)

    @staticmethod
    def valid_payload() -> dict[str, object]:
        return {
            "applicant_reference": "API-001",
            "person_age": 30,
            "person_income": 65_000,
            "person_emp_length": 5,
            "person_home_ownership": "RENT",
            "loan_amnt": 8_000,
            "loan_intent": "PERSONAL",
            "cb_person_cred_hist_length": 6,
            "cb_person_default_on_file": "N",
        }

    def post_score(self, payload: dict[str, object], key: str, request_id: uuid.UUID | None = None):
        headers = {"HTTP_X_API_KEY": key}
        if request_id:
            headers["HTTP_IDEMPOTENCY_KEY"] = str(request_id)
        return self.client.post(
            reverse("score-api"),
            data=json.dumps(payload),
            content_type="application/json",
            **headers,
        )

    @override_settings(
        SCORING_API_KEY="",
        SCORING_API_KEYS={"alpha": "alpha-secret", "beta": "beta-secret"},
        API_RATE_LIMIT_PER_MINUTE=20,
    )
    def test_idempotency_is_scoped_per_client_and_conflicts_on_changed_input(self) -> None:
        request_id = uuid.uuid4()
        first = self.post_score(self.valid_payload(), "alpha-secret", request_id)
        replay = self.post_score(self.valid_payload(), "alpha-secret", request_id)
        changed_payload = self.valid_payload()
        changed_payload["person_income"] = 70_000
        conflict = self.post_score(changed_payload, "alpha-secret", request_id)
        other_client = self.post_score(self.valid_payload(), "beta-secret", request_id)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay["Idempotent-Replay"], "true")
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(other_client.status_code, 200)
        self.assertEqual(
            set(AssessmentCase.objects.values_list("idempotency_namespace", flat=True)),
            {"api:alpha", "api:beta"},
        )
        self.assertEqual(PredictionAudit.objects.count(), 2)

    @override_settings(
        SCORING_API_KEY="",
        SCORING_API_KEYS={"alpha": "alpha-secret"},
        API_RATE_LIMIT_PER_MINUTE=1,
    )
    def test_invalid_auth_is_throttled_without_consuming_valid_client_budget(self) -> None:
        first_invalid = self.post_score(self.valid_payload(), "wrong-secret")
        throttled_invalid = self.post_score(self.valid_payload(), "still-wrong")
        valid = self.post_score(self.valid_payload(), "alpha-secret")

        self.assertEqual(first_invalid.status_code, 401)
        self.assertEqual(throttled_invalid.status_code, 429)
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(ApiRateLimitBucket.objects.count(), 2)

    @override_settings(
        SCORING_API_KEY="",
        SCORING_API_KEYS={"alpha": "alpha-secret"},
        API_RATE_LIMIT_PER_MINUTE=20,
    )
    def test_unknown_fields_and_material_out_of_domain_inputs_are_not_scored(self) -> None:
        unknown = self.valid_payload()
        unknown["loan_grade"] = "A"
        unknown_response = self.post_score(unknown, "alpha-secret")
        outside = self.valid_payload()
        outside["loan_amnt"] = 70_000
        outside_response = self.post_score(outside, "alpha-secret")

        self.assertEqual(unknown_response.status_code, 400)
        self.assertEqual(unknown_response.json()["fields"], ["loan_grade"])
        self.assertEqual(outside_response.status_code, 422)
        self.assertEqual(AssessmentCase.objects.count(), 0)
        self.assertEqual(PredictionAudit.objects.count(), 0)

    @override_settings(AUDIT_HMAC_KEY="new-hmac-key", AUDIT_HMAC_KEYS=["new-hmac-key", "old-hmac-key"])
    def test_retained_audit_hmac_key_preserves_exact_reference_lookup_during_rotation(self) -> None:
        reference = "ROTATE-001"
        old_digest = hmac.new(
            b"old-hmac-key",
            reference.casefold().encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        case = make_case(reference=reference)
        AssessmentCase.objects.filter(pk=case.pk).update(applicant_reference_digest=old_digest)

        self.assertIn(old_digest, services.reference_digests(reference))
        self.assertTrue(
            AssessmentCase.objects.filter(
                applicant_reference_digest__in=services.reference_digests(reference)
            ).exists()
        )


class DurableBatchWorkflowTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.bundle = load_joblib(services.MODEL_PATH)

    def setUp(self) -> None:
        self.model_loader = patch("app.services.load_model_bundle", return_value=self.bundle)
        self.model_loader.start()
        self.addCleanup(self.model_loader.stop)

    @staticmethod
    def csv_payload(reference: str = "QUEUE-001") -> bytes:
        return (
            "applicant_reference,person_age,person_income,person_emp_length,"
            "person_home_ownership,loan_amnt,loan_intent,"
            "cb_person_cred_hist_length,cb_person_default_on_file\n"
            f"{reference},30,65000,5,RENT,8000,PERSONAL,6,N\n"
        ).encode()

    @override_settings(LOGIN_REQUIRED=True, LOCAL_DEMO_MODE=False, BATCH_PROCESS_INLINE=False)
    def test_upload_is_encrypted_at_rest_then_worker_persists_rows_and_erases_payload(self) -> None:
        user = get_user_model().objects.create_user("batch-owner", password="test-password")
        add_to_group(user, "Analysts")
        self.client.force_login(user)
        raw_payload = self.csv_payload()
        response = self.client.post(
            reverse("batch-upload"),
            {"file": SimpleUploadedFile("queued.csv", raw_payload, content_type="text/csv")},
        )
        self.assertEqual(response.status_code, 302)
        batch = BatchAssessment.objects.get()
        self.assertEqual(batch.status, BatchAssessment.Status.PENDING)
        self.assertEqual(batch.upload_payload, raw_payload)
        with connection.cursor() as cursor:
            cursor.execute("SELECT upload_payload FROM app_batchassessment WHERE id = %s", [batch.id.hex])
            stored = bytes(cursor.fetchone()[0])
        self.assertNotIn(b"QUEUE-001", stored)

        process_batch(batch.id)
        batch.refresh_from_db()
        row = batch.rows.get()
        self.assertEqual(batch.status, BatchAssessment.Status.COMPLETE)
        self.assertEqual(batch.attempts, 1)
        self.assertEqual(batch.upload_payload, b"")
        self.assertEqual(row.status, BatchRow.Status.SCORED)
        self.assertEqual(row.case.batch, batch)
        self.assertEqual(batch.results[0]["case_id"], str(row.case_id))

    def test_cancelled_pending_batch_does_not_retain_uploaded_applicant_data(self) -> None:
        batch = BatchAssessment.objects.create(
            file_name="cancel.csv",
            upload_payload=self.csv_payload("CANCEL-001"),
            cancel_requested=True,
        )

        process_batch(batch.id)

        batch.refresh_from_db()
        self.assertEqual(batch.status, BatchAssessment.Status.CANCELLED)
        self.assertEqual(batch.upload_payload, b"")
        self.assertIsNotNone(batch.completed_at)
        self.assertFalse(batch.rows.exists())

    @override_settings(BATCH_LEASE_SECONDS=30, BATCH_MAX_ATTEMPTS=3)
    def test_stale_worker_leases_are_requeued_or_fail_at_the_retry_limit(self) -> None:
        stale_time = timezone.now() - timedelta(minutes=5)
        retryable = BatchAssessment.objects.create(
            file_name="retryable.csv",
            status=BatchAssessment.Status.PROCESSING,
            attempts=1,
            heartbeat_at=stale_time,
            worker_token=uuid.uuid4(),
        )
        exhausted = BatchAssessment.objects.create(
            file_name="exhausted.csv",
            status=BatchAssessment.Status.PROCESSING,
            attempts=3,
            heartbeat_at=stale_time,
            worker_token=uuid.uuid4(),
        )

        self.assertEqual(recover_stale_batches(), 2)
        retryable.refresh_from_db()
        exhausted.refresh_from_db()
        self.assertEqual(retryable.status, BatchAssessment.Status.PENDING)
        self.assertIsNone(retryable.worker_token)
        self.assertEqual(exhausted.status, BatchAssessment.Status.FAILED)
        self.assertIsNotNone(exhausted.completed_at)
        self.assertIn("retry limit", exhausted.error_message)

    @override_settings(LOGIN_REQUIRED=True, LOCAL_DEMO_MODE=False, BATCH_PROCESS_INLINE=False)
    def test_cancel_and_retry_endpoints_enforce_batch_ownership(self) -> None:
        owner = get_user_model().objects.create_user("batch-a", password="test-password")
        outsider = get_user_model().objects.create_user("batch-b", password="test-password")
        add_to_group(owner, "Analysts")
        add_to_group(outsider, "Analysts")
        pending = BatchAssessment.objects.create(created_by=owner, file_name="pending.csv")
        failed = BatchAssessment.objects.create(
            created_by=owner,
            file_name="failed.csv",
            status=BatchAssessment.Status.FAILED,
            cancel_requested=True,
        )
        self.client.force_login(outsider)
        self.assertEqual(
            self.client.post(reverse("batch-cancel", args=[pending.id])).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(reverse("batch-retry", args=[failed.id])).status_code,
            404,
        )

        self.client.force_login(owner)
        self.assertRedirects(
            self.client.post(reverse("batch-cancel", args=[pending.id])),
            reverse("batch-detail", args=[pending.id]),
        )
        self.assertRedirects(
            self.client.post(reverse("batch-retry", args=[failed.id])),
            reverse("batch-detail", args=[failed.id]),
        )
        pending.refresh_from_db()
        failed.refresh_from_db()
        self.assertTrue(pending.cancel_requested)
        self.assertEqual(failed.status, BatchAssessment.Status.PENDING)
        self.assertFalse(failed.cancel_requested)

    def test_xlsx_archive_traversal_and_csv_formula_injection_are_blocked(self) -> None:
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("../unsafe.xml", "unsafe")
        upload = SimpleUploadedFile("unsafe.xlsx", archive_bytes.getvalue())
        with self.assertRaisesRegex(ValueError, "unsafe internal path"):
            services.read_batch_upload(upload)

        output = services.batch_results_csv(
            [
                {
                    "row": 2,
                    "applicant_reference": "=HYPERLINK(\"https://example.test\")",
                    "status": "invalid",
                    "errors": ["+malicious-formula"],
                    "warnings": [],
                }
            ]
        )
        parsed = next(csv.DictReader(io.StringIO(output)))
        self.assertTrue(parsed["applicant_reference"].startswith("'="))
        self.assertTrue(parsed["errors"].startswith("'+"))


class RetentionAndDeletionWorkflowTests(TestCase):
    @override_settings(ACCESS_LOG_RETENTION_DAYS=30)
    def test_retention_dry_run_then_confirm_preserves_every_held_record(self) -> None:
        old = timezone.now() - timedelta(days=100)
        ordinary_batch = BatchAssessment.objects.create(file_name="ordinary.csv")
        held_batch = BatchAssessment.objects.create(file_name="held.csv")
        ordinary = make_case(batch=ordinary_batch, reference="PURGE-ME", namespace="web:purge")
        held = make_case(
            batch=held_batch,
            reference="KEEP-ME",
            namespace="web:held",
            legal_hold=True,
        )
        ordinary_audit = make_audit(ordinary)
        held_audit = make_audit(held)
        ordinary_log = SensitiveDataAccessLog.objects.create(
            action="case_viewed", object_type="AssessmentCase", object_id=str(ordinary.id)
        )
        held_log = SensitiveDataAccessLog.objects.create(
            action="case_viewed", object_type="AssessmentCase", object_id=str(held.id)
        )
        AssessmentCase.objects.filter(id__in=[ordinary.id, held.id]).update(created_at=old)
        BatchAssessment.objects.filter(id__in=[ordinary_batch.id, held_batch.id]).update(created_at=old)
        PredictionAudit.objects.filter(id__in=[ordinary_audit.id, held_audit.id]).update(created_at=old)
        SensitiveDataAccessLog.objects.filter(id__in=[ordinary_log.id, held_log.id]).update(
            created_at=old
        )

        call_command("purge_old_cases", days=30)
        self.assertEqual(AssessmentCase.objects.count(), 2)
        call_command("purge_old_cases", days=30, confirm=True)

        self.assertFalse(AssessmentCase.objects.filter(pk=ordinary.pk).exists())
        self.assertFalse(BatchAssessment.objects.filter(pk=ordinary_batch.pk).exists())
        self.assertFalse(PredictionAudit.objects.filter(pk=ordinary_audit.pk).exists())
        self.assertFalse(SensitiveDataAccessLog.objects.filter(pk=ordinary_log.pk).exists())
        self.assertTrue(AssessmentCase.objects.filter(pk=held.pk, legal_hold=True).exists())
        self.assertTrue(BatchAssessment.objects.filter(pk=held_batch.pk).exists())
        self.assertTrue(PredictionAudit.objects.filter(pk=held_audit.pk).exists())
        self.assertTrue(SensitiveDataAccessLog.objects.filter(pk=held_log.pk).exists())

    def test_subject_deletion_is_exact_scoped_and_redacts_batch_copies(self) -> None:
        shared_request_id = uuid.uuid4()
        target = make_case(
            reference="Subject-001",
            request_id=shared_request_id,
            namespace="web:target",
        )
        unrelated = make_case(
            reference="OTHER-001",
            request_id=shared_request_id,
            namespace="web:other",
        )
        target_audit = make_audit(target)
        unrelated_audit = make_audit(unrelated)
        batch = BatchAssessment.objects.create(
            file_name="subjects.csv",
            results=[
                {"row": 2, "applicant_reference": "subject-001", "status": "scored"},
                {"row": 3, "applicant_reference": "OTHER-001", "status": "scored"},
            ],
        )
        target_row = BatchRow.objects.create(
            batch=batch,
            row_number=2,
            applicant_reference="Subject-001",
            reference_digest=services.reference_digest("Subject-001"),
            status=BatchRow.Status.SCORED,
            case=target,
            result={"case_id": str(target.id)},
        )

        call_command(
            "delete_subject_data",
            applicant_reference=" SUBJECT-001 ",
            requested_by="privacy-office",
        )
        self.assertTrue(AssessmentCase.objects.filter(pk=target.pk).exists())
        call_command(
            "delete_subject_data",
            applicant_reference=" SUBJECT-001 ",
            requested_by="privacy-office",
            confirm=True,
        )

        self.assertFalse(AssessmentCase.objects.filter(pk=target.pk).exists())
        self.assertFalse(PredictionAudit.objects.filter(pk=target_audit.pk).exists())
        self.assertTrue(AssessmentCase.objects.filter(pk=unrelated.pk).exists())
        self.assertTrue(PredictionAudit.objects.filter(pk=unrelated_audit.pk).exists())
        target_row.refresh_from_db()
        self.assertEqual(target_row.status, BatchRow.Status.REDACTED)
        self.assertEqual(target_row.applicant_reference, "")
        batch.refresh_from_db()
        self.assertEqual(batch.results[0]["status"], "redacted")
        self.assertEqual(batch.results[1]["applicant_reference"], "OTHER-001")
        receipt = DataDeletionReceipt.objects.get()
        self.assertEqual(receipt.requested_by, "privacy-office")
        self.assertEqual(receipt.deleted_cases, 1)
        self.assertEqual(receipt.deleted_audits, 1)

    def test_subject_deletion_is_blocked_by_legal_hold(self) -> None:
        held = make_case(reference="HELD-SUBJECT", legal_hold=True)
        with self.assertRaisesRegex(CommandError, "legal hold"):
            call_command(
                "delete_subject_data",
                applicant_reference="held-subject",
                requested_by="privacy-office",
                confirm=True,
            )
        self.assertTrue(AssessmentCase.objects.filter(pk=held.pk).exists())
        self.assertFalse(DataDeletionReceipt.objects.exists())


class OutcomeAndMonitoringWorkflowTests(TestCase):
    def setUp(self) -> None:
        self.actor = get_user_model().objects.create_user("outcome-owner", password="test-password")

    def temporary_csv(self, content: str) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            encoding="utf-8",
            newline="",
            delete=False,
        )
        with handle:
            handle.write(content)
        path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_outcome_import_is_dry_run_by_default_then_creates_event_on_confirm(self) -> None:
        case = make_case(created_by=self.actor)
        path = self.temporary_csv(
            "case_id,outcome,outcome_date,performance_window_end,as_of_date,"
            "exposure_at_default,loss_amount,source,source_reference,notes\n"
            f"{case.id},defaulted,2026-01-15,2025-12-31,2026-02-01,"
            "10000,2500,servicing,OUT-001,Confirmed arrears\n"
        )

        call_command("import_outcomes", str(path), actor=self.actor.username)
        self.assertFalse(CaseOutcome.objects.exists())
        call_command("import_outcomes", str(path), actor=self.actor.username, confirm=True)

        outcome = CaseOutcome.objects.get(case=case)
        self.assertEqual(outcome.outcome, CaseOutcome.Outcome.DEFAULTED)
        self.assertEqual(outcome.loss_amount, 2500)
        case.refresh_from_db()
        self.assertEqual(case.review_version, 1)
        self.assertEqual(case.review_events.get().event_type, CaseReviewEvent.EventType.OUTCOME)

    def test_outcome_import_rejects_the_whole_file_when_one_row_is_invalid(self) -> None:
        valid_case = make_case(created_by=self.actor, reference="OUT-VALID")
        invalid_case = make_case(created_by=self.actor, reference="OUT-INVALID")
        path = self.temporary_csv(
            "case_id,outcome,outcome_date,performance_window_end,as_of_date,"
            "exposure_at_default,loss_amount,source\n"
            f"{valid_case.id},performing,2025-01-01,2025-12-31,2026-01-01,10000,0,servicing\n"
            f"{invalid_case.id},defaulted,2026-01-01,2025-12-31,2026-02-01,100,200,servicing\n"
        )

        with self.assertRaisesRegex(CommandError, "rejected atomically"):
            call_command("import_outcomes", str(path), actor=self.actor.username, confirm=True)
        self.assertFalse(CaseOutcome.objects.exists())
        self.assertFalse(CaseReviewEvent.objects.exists())

    @override_settings(LOCAL_DEMO_MODE=True)
    def test_authenticated_monitoring_run_persists_alert_and_outcome_evidence(self) -> None:
        manifest = json.loads(services.MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        performing_case = make_case(
            created_by=self.actor,
            reference="MON-PERFORMING",
            model_version=manifest["model_version"],
            release_id=manifest["model_sha256"],
            probability=0.1,
        )
        default_case = make_case(
            created_by=self.actor,
            reference="MON-DEFAULT",
            model_version=manifest["model_version"],
            release_id=manifest["model_sha256"],
            probability=0.8,
        )
        for case, outcome_name, loss in (
            (performing_case, CaseOutcome.Outcome.PERFORMING, 0),
            (default_case, CaseOutcome.Outcome.DEFAULTED, 2500),
        ):
            CaseOutcome.objects.create(
                case=case,
                recorded_by=self.actor,
                outcome=outcome_name,
                outcome_date=date(2026, 1, 1),
                performance_window_end=date(2025, 12, 31),
                as_of_date=date(2026, 2, 1),
                exposure_at_default=10_000,
                loss_amount=loss,
                source="servicing",
            )
        monitoring_rows = "".join(
            "65000,5,8000,6,RENT,PERSONAL,N\n" for _ in range(100)
        )
        path = self.temporary_csv(
            "person_income,person_emp_length,loan_amnt,cb_person_cred_hist_length,"
            "person_home_ownership,loan_intent,cb_person_default_on_file\n"
            + monitoring_rows
        )

        call_command(
            "record_monitoring_run",
            str(path),
            window_start="2026-07-01",
            window_end="2026-07-31",
            as_of="2026-08-01",
            owner="Model Risk",
        )
        self.assertFalse(MonitoringRun.objects.exists())
        call_command(
            "record_monitoring_run",
            str(path),
            window_start="2026-07-01",
            window_end="2026-07-31",
            as_of="2026-08-01",
            owner="Model Risk",
            confirm=True,
        )

        run = MonitoringRun.objects.get()
        self.assertEqual(run.status, MonitoringRun.Status.ALERT)
        self.assertEqual(run.sample_size, 100)
        self.assertEqual(run.mature_outcome_count, 2)
        self.assertGreaterEqual(len(run.alerts), 1)
        self.assertTrue(all(alert["status"] == "drift" for alert in run.alerts))
        performance = run.metrics["performance_rows"]
        self.assertEqual(performance[0]["mature_outcomes"], 2)
        self.assertEqual(performance[0]["evidence_status"], "insufficient")


@override_settings(LOGIN_REQUIRED=False, LOCAL_DEMO_MODE=True)
class PolicyApprovalAccountabilityTests(TestCase):
    def test_anonymous_demo_user_cannot_decide_an_existing_scenario(self) -> None:
        scenario = PolicyScenario.objects.create(
            name="Anonymous approval test",
            assumptions={},
            results={},
            model_version="2.2.0",
        )

        response = self.client.post(
            reverse("policy-scenario-decision", args=[scenario.id]),
            {
                "decision": PolicyScenario.Status.APPROVED,
                "reason": "Anonymous approval must never be accepted.",
            },
        )

        self.assertEqual(response.status_code, 403)
        scenario.refresh_from_db()
        self.assertEqual(scenario.status, PolicyScenario.Status.DRAFT)
        self.assertFalse(PolicyScenarioEvent.objects.exists())

    def test_authenticated_administrator_decision_is_attributed_and_immutable(self) -> None:
        admin = get_user_model().objects.create_superuser(
            "policy-admin", email="admin@example.test", password="test-password"
        )
        scenario = PolicyScenario.objects.create(
            name="Reviewed scenario",
            assumptions={},
            results={},
            model_version="2.2.0",
        )
        self.client.force_login(admin)

        response = self.client.post(
            reverse("policy-scenario-decision", args=[scenario.id]),
            {
                "decision": PolicyScenario.Status.APPROVED,
                "reason": "Independent model-risk approval completed.",
            },
        )

        self.assertRedirects(response, reverse("business-policy"))
        scenario.refresh_from_db()
        self.assertEqual(scenario.status, PolicyScenario.Status.APPROVED)
        self.assertEqual(scenario.approved_by, admin)
        event = scenario.events.get()
        self.assertEqual(event.actor, admin)
        self.assertEqual(event.action, PolicyScenarioEvent.Action.APPROVED)
        event.reason = "Rewrite"
        with self.assertRaisesRegex(ValidationError, "immutable"):
            event.save()
