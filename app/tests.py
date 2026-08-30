from __future__ import annotations

import io
import json
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
from joblib import load as load_joblib
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import connection
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from app import services
from app.forms import ApplicantAssessmentForm
from app.models import AssessmentCase, BatchAssessment, PredictionAudit, SensitiveDataAccessLog
from src.train_model import (
    EXCLUDED_LENDER_ASSIGNED_FEATURES,
    FEATURES,
    build_threshold_table,
    choose_business_threshold,
    evaluate_predictions,
    load_credit_data,
)


def test_model_bundle() -> dict:
    """A direct test fixture; production loading still verifies releases."""
    return load_joblib(services.MODEL_PATH)


class ApplicantAssessmentFormTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.bundle = test_model_bundle()

    def valid_payload(self) -> dict[str, str]:
        return {
            "person_age": "30",
            "person_income": "65000",
            "person_emp_length": "5",
            "person_home_ownership": "RENT",
            "loan_amnt": "8000",
            "loan_intent": "PERSONAL",
            "cb_person_cred_hist_length": "6",
            "cb_person_default_on_file": "N",
        }

    def test_form_uses_only_application_time_features(self) -> None:
        form = ApplicantAssessmentForm(bundle=self.bundle)
        self.assertNotIn("loan_grade", form.fields)
        self.assertNotIn("loan_int_rate", form.fields)
        self.assertEqual(EXCLUDED_LENDER_ASSIGNED_FEATURES, ["loan_int_rate", "loan_grade"])

    def test_valid_application(self) -> None:
        form = ApplicantAssessmentForm(self.valid_payload(), bundle=self.bundle)
        self.assertTrue(form.is_valid(), form.errors)

    def test_new_form_is_blank_unless_demo_is_requested(self) -> None:
        form = ApplicantAssessmentForm(bundle=self.bundle)
        self.assertIsNone(form["person_income"].value())
        self.assertIsNone(form["person_home_ownership"].value())
        demo = ApplicantAssessmentForm(bundle=self.bundle, use_demo=True)
        self.assertEqual(
            demo["person_income"].value(),
            round(self.bundle["feature_reference"]["numeric_medians"]["person_income"]),
        )
        self.assertEqual(demo["person_home_ownership"].value(), "RENT")

    def test_zero_income_is_rejected(self) -> None:
        payload = self.valid_payload()
        payload["person_income"] = "0"
        form = ApplicantAssessmentForm(payload, bundle=self.bundle)
        self.assertFalse(form.is_valid())
        self.assertIn("person_income", form.errors)

    def test_impossible_employment_history_is_rejected(self) -> None:
        payload = self.valid_payload()
        payload.update({"person_age": "20", "person_emp_length": "10"})
        form = ApplicantAssessmentForm(payload, bundle=self.bundle)
        self.assertFalse(form.is_valid())
        self.assertIn("person_emp_length", form.errors)

    def test_extreme_but_valid_values_generate_warning(self) -> None:
        payload = self.valid_payload()
        payload.update(
            {
                "person_age": "90",
                "person_income": "10000",
                "loan_amnt": "10000",
                "person_emp_length": "1",
                "cb_person_cred_hist_length": "1",
            }
        )
        form = ApplicantAssessmentForm(payload, bundle=self.bundle)
        self.assertTrue(form.is_valid(), form.errors)
        self.assertTrue(form.distribution_warnings())


class ServiceTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.bundle = test_model_bundle()

    @override_settings(LOCAL_DEMO_MODE=False, DATA_PROVENANCE_VERIFIED=False)
    def test_unreleased_model_is_rejected_by_runtime_loader(self) -> None:
        services.load_model_bundle.cache_clear()
        try:
            with self.assertRaises(services.ArtifactIntegrityError):
                services.load_model_bundle()
        finally:
            services.load_model_bundle.cache_clear()

    @override_settings(LOCAL_DEMO_MODE=True)
    def test_local_demo_mode_loads_hash_checked_bundled_model(self) -> None:
        services.load_model_bundle.cache_clear()
        try:
            bundle = services.load_model_bundle()
            self.assertEqual(bundle["model_version"], "2.2.0")
        finally:
            services.load_model_bundle.cache_clear()

    @override_settings(LOCAL_DEMO_MODE=True)
    @patch("app.services._sha256", return_value="0" * 64)
    def test_local_demo_mode_rejects_model_hash_mismatch(self, _sha256) -> None:
        services.load_model_bundle.cache_clear()
        try:
            with self.assertRaisesRegex(
                services.ArtifactIntegrityError,
                "integrity verification failed",
            ):
                services.load_model_bundle()
        finally:
            services.load_model_bundle.cache_clear()

    def test_bundle_has_governance_metadata(self) -> None:
        for key in (
            "model_version",
            "trained_at_utc",
            "data_sha256",
            "git_commit",
            "split_sizes",
            "runtime_versions",
            "predictor",
            "model_name",
        ):
            self.assertIn(key, self.bundle)

    def test_legacy_risk_bands_keep_manual_review_reachable(self) -> None:
        category, _, _ = services.risk_category(0.25, self.bundle)
        self.assertEqual(category, "Medium")

    def test_application_frame_matches_model_contract(self) -> None:
        cleaned = {
            "person_age": 30,
            "person_income": 65000,
            "person_emp_length": 5.0,
            "person_home_ownership": "RENT",
            "loan_amnt": 8000,
            "loan_intent": "PERSONAL",
            "cb_person_cred_hist_length": 6,
            "cb_person_default_on_file": "N",
        }
        _, frame = services.application_from_cleaned_data(self.bundle, cleaned)
        self.assertEqual(frame.columns.tolist(), FEATURES)
        self.assertAlmostEqual(frame.iloc[0]["loan_percent_income"], 0.1231)

    @override_settings(LOCAL_DEMO_MODE=True)
    def test_artifact_allowlist_blocks_unknown_files(self) -> None:
        self.assertIsNone(services.report_artifact_path("../../README.md"))
        self.assertIsNotNone(services.report_artifact_path("calibration_curve.png"))

    def test_threshold_selection_uses_lowest_cost(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.45, 0.9])
        table = build_threshold_table(y_true, probabilities)
        threshold = choose_business_threshold(table)
        selected = table.loc[table["threshold"].eq(threshold)].iloc[0]
        self.assertEqual(selected["business_cost"], table["business_cost"].min())

    def test_metrics_include_probability_quality(self) -> None:
        metrics = evaluate_predictions(
            pd.Series([0, 0, 1, 1]),
            np.array([0.05, 0.2, 0.8, 0.95]),
        )
        self.assertIn("average_precision", metrics)
        self.assertIn("brier_score", metrics)

    def test_cleaned_training_data_removes_duplicates_and_outliers(self) -> None:
        data = load_credit_data()
        self.assertEqual(data.duplicated().sum(), 0)
        self.assertEqual(int((data["person_age"] > 100).sum()), 0)
        self.assertEqual(int((data["person_emp_length"] > 60).sum()), 0)

    @override_settings(AUDIT_HMAC_KEY="test-hmac-key")
    def test_feature_digest_is_keyed(self) -> None:
        application = pd.DataFrame([{"person_income": 65000}])
        digest = services.feature_digest(application)
        plain = __import__("hashlib").sha256(
            json.dumps(
                application.iloc[0].to_dict(),
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.assertNotEqual(digest, plain)

    def test_business_economics_returns_a_recommended_threshold(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "threshold": 0.2,
                    "false_negatives": 10,
                    "false_positives": 20,
                    "true_negatives": 70,
                    "true_positives": 30,
                    "precision": 0.6,
                    "recall": 0.75,
                },
                {
                    "threshold": 0.5,
                    "false_negatives": 20,
                    "false_positives": 5,
                    "true_negatives": 85,
                    "true_positives": 20,
                    "precision": 0.8,
                    "recall": 0.5,
                },
            ]
        )
        result = services.business_economics(
            table,
            average_exposure=10000,
            loss_given_default=0.6,
            annual_margin=0.08,
            review_cost=35,
        )
        self.assertIn("estimated_total_cost", result["table"])
        self.assertIn("threshold", result["recommended"])


class DashboardViewTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model_loader = patch("app.services.load_model_bundle", return_value=test_model_bundle())
        self.data_loader = patch("app.services.load_credit_data", return_value=load_credit_data())
        self.model_loader.start()
        self.data_loader.start()
        self.addCleanup(self.model_loader.stop)
        self.addCleanup(self.data_loader.stop)

    def valid_payload(self) -> dict[str, str]:
        return {
            "person_age": "30",
            "person_income": "65000",
            "person_emp_length": "5",
            "person_home_ownership": "RENT",
            "loan_amnt": "8000",
            "loan_intent": "PERSONAL",
            "cb_person_cred_hist_length": "6",
            "cb_person_default_on_file": "N",
        }

    @override_settings(LOCAL_DEMO_MODE=True)
    def test_health_and_readiness(self) -> None:
        health = self.client.get(reverse("health"))
        ready = self.client.get(reverse("readiness"))
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")

    @patch("app.services.assessment_result")
    def test_get_assessment_does_not_score(self, assessment_result) -> None:
        response = self.client.get(reverse("assessment"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["has_result"])
        assessment_result.assert_not_called()
        self.assertContains(response, "Ready for an application")
        self.assertNotContains(response, "Model Insights")
        self.assertNotContains(response, "Threshold Analysis")
        self.assertNotContains(response, "Model v")

    def test_home_page_is_the_client_assessment(self) -> None:
        response = self.client.get(reverse("assessment"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Show assessment")
        self.assertContains(response, "People stay in control")

    @override_settings(LOCAL_DEMO_MODE=True)
    def test_local_demo_mode_is_clearly_labelled(self) -> None:
        response = self.client.get(reverse("assessment"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Local demo")

    @override_settings(LOCAL_DEMO_MODE=True)
    def test_operational_pages_are_available_in_local_mode(self) -> None:
        for route_name in (
            "case-list",
            "batch-upload",
            "monitoring",
            "business-policy",
            "reports",
            "api-docs",
        ):
            response = self.client.get(reverse(route_name))
            self.assertEqual(response.status_code, 200)

    def test_invalid_post_shows_errors_without_audit_record(self) -> None:
        payload = self.valid_payload()
        payload["person_income"] = "0"
        response = self.client.post(reverse("assessment"), payload)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["has_result"])
        self.assertEqual(PredictionAudit.objects.count(), 0)
        self.assertContains(response, "Ensure this value is greater than or equal to 1")

    @patch("app.services.applicant_explanations")
    def test_valid_post_scores_and_writes_privacy_preserving_audit(self, explanations) -> None:
        explanations.return_value = (
            [
                {
                    "factor": "Income",
                    "detail": "Test explanation",
                    "impact": "Reduces risk",
                    "class": "positive",
                }
            ],
            "Test method",
            "model",
        )
        response = self.client.post(reverse("assessment"), self.valid_payload())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["has_result"])
        audit = PredictionAudit.objects.get()
        self.assertEqual(len(audit.feature_digest), 64)
        self.assertEqual(audit.digest_version, "hmac-sha256-v1")
        self.assertEqual(audit.model_version, "2.2.0")
        self.assertEqual(AssessmentCase.objects.count(), 1)
        self.assertEqual(SensitiveDataAccessLog.objects.filter(action="case_created").count(), 1)
        with connection.cursor() as cursor:
            cursor.execute("SELECT application_data FROM app_assessmentcase LIMIT 1")
            stored_value = cursor.fetchone()[0]
        self.assertNotIn("person_income", stored_value)
        self.assertNotIn("65000", stored_value)

    @override_settings(LOCAL_DEMO_MODE=True)
    def test_report_downloads_and_allowlist(self) -> None:
        self.assertEqual(self.client.get(reverse("download-summary-csv")).status_code, 200)
        self.assertEqual(self.client.get(reverse("download-summary-pdf")).status_code, 200)
        self.assertEqual(
            self.client.get(reverse("report-artifact", args=["calibration_curve.png"])).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(reverse("report-artifact", args=["unknown.txt"])).status_code,
            404,
        )

    def test_api_reference_pdf_download(self) -> None:
        response = self.client.get(reverse("download-api-reference-pdf"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("aegis-credit-scoring-api-reference.pdf", response["Content-Disposition"])
        self.assertTrue(response.content.startswith(b"%PDF-"))

    def test_model_manifest_matches_bundle(self) -> None:
        manifest = json.loads((services.PROJECT_ROOT / "models" / "model_manifest.json").read_text())
        bundle = services.load_model_bundle()
        self.assertEqual(manifest["model_version"], bundle["model_version"])
        self.assertEqual(manifest["data_sha256"], bundle["data_sha256"])

    @override_settings(SCORING_API_KEY="test-api-key")
    def test_scoring_api_requires_authentication(self) -> None:
        response = self.client.post(
            reverse("score-api"),
            data=json.dumps(self.valid_payload()),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 401)

    @override_settings(SCORING_API_KEY="test-api-key")
    def test_scoring_api_validates_payload(self) -> None:
        payload = self.valid_payload()
        payload["person_income"] = "0"
        response = self.client.post(
            reverse("score-api"),
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_API_KEY="test-api-key",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("person_income", response.json()["fields"])

    @override_settings(SCORING_API_KEY="test-api-key")
    def test_scoring_api_returns_versioned_result(self) -> None:
        request_id = str(uuid.uuid4())
        response = self.client.post(
            reverse("score-api"),
            data=json.dumps(self.valid_payload()),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-api-key",
            HTTP_IDEMPOTENCY_KEY=request_id,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], "2.2.0")
        self.assertEqual(response.json()["api_client_id"], "legacy")
        self.assertIn("probability", response.json())
        replay = self.client.post(
            reverse("score-api"),
            data=json.dumps(self.valid_payload()),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-api-key",
            HTTP_IDEMPOTENCY_KEY=request_id,
        )
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay["Idempotent-Replay"], "true")
        self.assertEqual(AssessmentCase.objects.filter(source="api").count(), 1)

    @override_settings(SCORING_API_KEY="rate-test-key", API_RATE_LIMIT_PER_MINUTE=1)
    def test_scoring_api_rate_limit(self) -> None:
        cache.clear()
        first = self.client.post(
            reverse("score-api"),
            data=json.dumps(self.valid_payload()),
            content_type="application/json",
            HTTP_X_API_KEY="rate-test-key",
        )
        second = self.client.post(
            reverse("score-api"),
            data=json.dumps(self.valid_payload()),
            content_type="application/json",
            HTTP_X_API_KEY="rate-test-key",
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second["Retry-After"], "60")

    @override_settings(SCORING_API_KEY="test-api-key")
    def test_scoring_api_rejects_invalid_idempotency_key(self) -> None:
        response = self.client.post(
            reverse("score-api"),
            data=json.dumps(self.valid_payload()),
            content_type="application/json",
            HTTP_X_API_KEY="test-api-key",
            HTTP_IDEMPOTENCY_KEY="not-a-uuid",
        )
        self.assertEqual(response.status_code, 400)

    def test_batch_csv_separates_valid_and_invalid_rows(self) -> None:
        content = (
            "applicant_reference,person_age,person_income,person_emp_length,"
            "person_home_ownership,loan_amnt,loan_intent,"
            "cb_person_cred_hist_length,cb_person_default_on_file\n"
            "GOOD-1,30,65000,5,RENT,8000,PERSONAL,6,N\n"
            "BAD-1,30,0,5,RENT,8000,PERSONAL,6,N\n"
        )
        upload = SimpleUploadedFile(
            "applications.csv",
            content.encode(),
            content_type="text/csv",
        )
        response = self.client.post(reverse("batch-upload"), {"file": upload})
        self.assertEqual(response.status_code, 302)
        batch = BatchAssessment.objects.get()
        self.assertEqual(batch.valid_rows, 1)
        self.assertEqual(batch.invalid_rows, 1)

    def test_override_requires_reason(self) -> None:
        response = self.client.post(reverse("assessment"), self.valid_payload())
        case = AssessmentCase.objects.latest("created_at")
        response = self.client.post(
            reverse("case-detail", args=[case.id]),
            {
                "form_action": "review",
                "review-expected_version": case.review_version,
                "review-status": AssessmentCase.Status.IN_REVIEW,
                "review-reviewer_notes": "",
                "review-override_decision": AssessmentCase.OverrideDecision.MANUAL,
                "review-override_reason": "",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Explain why")

    @override_settings(LOGIN_REQUIRED=True)
    def test_protected_page_requires_login_and_role(self) -> None:
        response = self.client.get(reverse("case-list"))
        self.assertEqual(response.status_code, 302)
        user = get_user_model().objects.create_user("analyst", password="safe-test-password")
        group, _ = Group.objects.get_or_create(name="Analysts")
        user.groups.add(group)
        self.client.login(username="analyst", password="safe-test-password")
        self.assertEqual(self.client.get(reverse("case-list")).status_code, 200)
        self.assertEqual(self.client.get(reverse("monitoring")).status_code, 403)
