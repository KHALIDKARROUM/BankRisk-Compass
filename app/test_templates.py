from __future__ import annotations

import uuid
from types import SimpleNamespace

from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser, Group
from django.core.paginator import Paginator
from django.template.loader import render_to_string
from django.test import RequestFactory, TestCase, override_settings
from django.urls import resolve, reverse
from django.utils import timezone
from joblib import load as load_joblib

from . import services
from .context_processors import product_shell
from .forms import ApplicantAssessmentForm, BatchUploadForm, BusinessEconomicsForm


class _Batch:
    id = uuid.uuid4()
    file_name = "demo-applications.csv"
    status = "complete"
    valid_rows = 1
    invalid_rows = 0
    results = [
        {
            "row": 2,
            "applicant_reference": "DEMO-001",
            "status": "scored",
            "probability": 0.2,
            "risk_category": "Low",
            "recommended_next_step": "Continue with standard review",
            "warnings": ["Income is outside the usual range; verify it."],
            "errors": [],
        }
    ]

    @staticmethod
    def get_status_display() -> str:
        return "Complete"


class _Case:
    id = uuid.uuid4()
    applicant_reference = "DEMO-CASE"
    created_at = timezone.now()
    model_version = "test-version"
    source = "web"
    probability_percent = "20.0%"
    risk_category = "Low"
    recommendation = "Continue with standard review"
    threshold = 0.21
    reviewed_at = None
    override_decision = ""
    legal_hold = False
    warnings: list[str] = []
    explanation_method = "Test explanation"
    explanation_rows: list[dict[str, str]] = []
    reviewer_notes = ""
    override_reason = ""
    assigned_to = None

    @staticmethod
    def get_status_display() -> str:
        return "New"


@override_settings(LOCAL_DEMO_MODE=True, LOGIN_REQUIRED=False)
class ProductShellTemplateTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.bundle = load_joblib(services.MODEL_PATH)

    def request(self, path: str = "/"):
        request = RequestFactory().get(path)
        request.user = AnonymousUser()
        request.resolver_match = resolve(path)
        return request

    def shell_context(self) -> dict[str, object]:
        return {
            "pages": [],
            "active_page": "",
            "display_date": "Jan 1, 2026",
            "local_demo_mode": True,
        }

    def test_context_processor_filters_navigation_by_role(self) -> None:
        analyst = get_user_model().objects.create_user("template-analyst")
        analyst_group, _ = Group.objects.get_or_create(name="Analysts")
        analyst.groups.add(analyst_group)
        request = self.request("/cases/")
        request.user = analyst

        with override_settings(LOGIN_REQUIRED=True, LOCAL_DEMO_MODE=False, DEBUG=False):
            context = product_shell(request)

        keys = {item["key"] for item in context["product_nav_items"]}
        self.assertTrue({"assessment", "cases", "batch", "api"}.issubset(keys))
        self.assertTrue(
            {"overview", "monitoring", "insights", "threshold", "business", "reports"}.isdisjoint(keys)
        )
        self.assertEqual(context["workspace_badge_label"], "Controlled workspace")

    def test_legal_navigation_is_limited_to_case_and_monitoring_work(self) -> None:
        legal = get_user_model().objects.create_user("template-legal")
        legal_group, _ = Group.objects.get_or_create(name="Legal Officers")
        legal.groups.add(legal_group)
        request = self.request("/monitoring/")
        request.user = legal

        with override_settings(LOGIN_REQUIRED=True):
            context = product_shell(request)

        self.assertEqual(
            {item["key"] for item in context["product_nav_items"]},
            {"cases", "monitoring"},
        )

    def test_context_processor_labels_local_demo_consistently(self) -> None:
        context = product_shell(self.request("/monitoring/"))
        self.assertTrue(context["product_shell_ready"])
        self.assertEqual(context["workspace_badge_label"], "Local demo")
        self.assertEqual(context["footer_message"], "Monitoring indicators require an owner, a freshness check, and documented follow-up.")

    def test_assessment_hints_are_identifiable_and_age_is_transparent(self) -> None:
        context = self.shell_context()
        context.update(
            {
                "active_page": "assessment",
                "assessment_form": ApplicantAssessmentForm(bundle=self.bundle),
                "has_result": False,
            }
        )
        html = render_to_string("app/assessment.html", context, request=self.request("/"))
        self.assertIn('id="id_person_income_hint"', html)
        self.assertIn("Age is excluded from the model score", html)
        self.assertIn("Complete the required fields", html)

    def test_django_help_text_ids_have_matching_elements(self) -> None:
        batch_context = self.shell_context()
        batch_context.update(
            {"active_page": "batch", "batch_form": BatchUploadForm(), "recent_batches": []}
        )
        batch_html = render_to_string(
            "app/batch_upload.html", batch_context, request=self.request("/batch/")
        )
        self.assertIn('aria-describedby="id_file_helptext"', batch_html)
        self.assertIn('id="id_file_helptext"', batch_html)

        policy_context = self.shell_context()
        policy_context.update(
            {
                "active_page": "business",
                "economics_form": BusinessEconomicsForm(),
                "recommended_display": {},
                "model_threshold": 0.21,
                "economics_table": SimpleNamespace(columns=[], rows=[]),
            }
        )
        policy_html = render_to_string(
            "app/business_policy.html",
            policy_context,
            request=self.request("/business-policy/"),
        )
        self.assertIn('aria-describedby="id_loss_given_default_helptext"', policy_html)
        self.assertIn('id="id_loss_given_default_helptext"', policy_html)

    def test_batch_warnings_are_visible_on_detail_page(self) -> None:
        context = self.shell_context()
        context.update(
            {"active_page": "batch", "batch": _Batch(), "batch_warning_count": 1}
        )
        html = render_to_string(
            "app/batch_detail.html", context, request=self.request(f"/batch/{_Batch.id}/")
        )
        self.assertIn("Verify input", html)
        self.assertIn("Income is outside the usual range", html)
        self.assertIn("need input verification", html)

    def test_batch_detail_accepts_durable_row_records(self) -> None:
        context = self.shell_context()
        context.update(
            {
                "active_page": "batch",
                "batch": _Batch(),
                "batch_rows": [
                    SimpleNamespace(
                        row_number=2,
                        applicant_reference="DURABLE-001",
                        status="scored",
                        result={
                            "probability": 0.314,
                            "risk_category": "Medium",
                            "recommended_next_step": "Route to manual review",
                        },
                        warnings=[],
                        errors=[],
                        case=None,
                    )
                ],
            }
        )
        html = render_to_string(
            "app/batch_detail.html", context, request=self.request(f"/batch/{_Batch.id}/")
        )
        self.assertIn("0.314", html)
        self.assertIn("Medium", html)
        self.assertIn("Route to manual review", html)

    def test_unreviewed_case_does_not_claim_a_human_decision(self) -> None:
        context = self.shell_context()
        context.update(
            {
                "active_page": "cases",
                "case": _Case(),
                "application_rows": [],
                "review_form": [],
                "can_review": False,
            }
        )
        html = render_to_string(
            "app/case_detail.html", context, request=self.request(f"/cases/{_Case.id}/")
        )
        self.assertIn("Not yet reviewed", html)
        self.assertNotIn("Effective human decision", html)

    def test_case_queue_supports_paginated_results(self) -> None:
        page_obj = Paginator([_Case(), _Case()], 1).get_page(1)
        context = self.shell_context()
        context.update(
            {
                "active_page": "cases",
                "cases": [],
                "page_obj": page_obj,
                "query": "",
                "selected_status": "",
                "selected_risk": "",
                "status_choices": [],
                "risk_choices": [],
            }
        )
        html = render_to_string(
            "app/case_list.html", context, request=self.request("/cases/")
        )
        self.assertIn("Showing 1–1 of 2 cases", html)
        self.assertIn("Page 1 of 2", html)
        self.assertIn("page=2", html)
        self.assertIn("Assigned to", html)

    def test_monitoring_unknown_freshness_is_explicit(self) -> None:
        context = self.shell_context()
        context.update(
            {
                "active_page": "monitoring",
                "total_scores": 0,
                "open_cases": 0,
                "reviewed_cases": 0,
                "override_rate": "0.0%",
                "by_risk": [],
                "by_source": [],
                "daily": [],
                "drift_table": SimpleNamespace(columns=[], rows=[]),
            }
        )
        html = render_to_string(
            "app/monitoring.html", context, request=self.request("/monitoring/")
        )
        self.assertIn("No drift report is available", html)
        self.assertIn("Feature drift scores", html)

    def test_monitoring_alert_exposes_follow_up_workflow(self) -> None:
        run_id = uuid.uuid4()
        context = self.shell_context()
        context.update(
            {
                "active_page": "monitoring",
                "product_role": "reviewer",
                "total_scores": 1,
                "open_cases": 1,
                "reviewed_cases": 0,
                "override_rate": "0.0%",
                "by_risk": [],
                "by_source": [],
                "daily": [],
                "monitoring_status": "fresh",
                "drift_table": SimpleNamespace(columns=["Feature"], rows=[["income"]]),
                "performance_table": SimpleNamespace(columns=[], rows=[]),
                "latest_monitoring_run": SimpleNamespace(
                    id=run_id,
                    status="alert",
                    alerts=["Population stability threshold exceeded"],
                    owner="Model Risk",
                    get_status_display=lambda: "Alert",
                ),
                "monitoring_acknowledgements": [],
            }
        )
        html = render_to_string(
            "app/monitoring.html", context, request=self.request("/monitoring/")
        )
        self.assertIn("latest monitoring run contains alerts", html)
        self.assertIn(f'/monitoring/{run_id}/acknowledge/', html)
        self.assertIn("No follow-up has been recorded", html)

    def test_policy_scenario_history_and_admin_decision_controls_render(self) -> None:
        scenario_id = uuid.uuid4()
        context = self.shell_context()
        context.update(
            {
                "active_page": "business",
                "product_role": "admin",
                "economics_form": BusinessEconomicsForm(),
                "recommended_display": {},
                "model_threshold": 0.21,
                "economics_table": SimpleNamespace(columns=[], rows=[]),
                "scenario_is_illustrative": True,
                "policy_scenarios": [
                    SimpleNamespace(
                        id=scenario_id,
                        name="Capacity constrained",
                        version=2,
                        model_version="2.2.0",
                        status="draft",
                        results={"recommended": {"threshold": 0.27}},
                        created_at=timezone.now(),
                        created_by=None,
                        approved_by=None,
                        approved_at=None,
                        get_status_display=lambda: "Draft",
                    )
                ],
            }
        )
        html = render_to_string(
            "app/business_policy.html",
            context,
            request=self.request("/business-policy/"),
        )
        self.assertIn("Scenario outputs are illustrative", html)
        self.assertIn("Save draft scenario", html)
        self.assertIn("Capacity constrained", html)
        self.assertIn(f'/business-policy/scenarios/{scenario_id}/decision/', html)

    def test_anonymous_local_demo_cannot_approve_policy_scenarios(self) -> None:
        response = self.client.post(
            reverse("policy-scenario-decision", args=[uuid.uuid4()]),
            {"decision": "approved", "reason": "Independent approval recorded."},
        )

        self.assertEqual(response.status_code, 403)

    def test_reports_have_page_structure_and_artifact_links(self) -> None:
        empty_table = SimpleNamespace(columns=[], rows=[])
        context = self.shell_context()
        context.update(
            {
                "active_page": "reports",
                "report_summary": {
                    "model_summary": [],
                    "dataset_summary": [],
                    "threshold_summary": [],
                    "confusion": [],
                    "business_recommendation": "Test recommendation",
                },
                "business_report_html": "<p>Test report</p>",
                "classification_report": "Test classification report",
                "final_metrics_table": empty_table,
                "fairness_table": empty_table,
                "artifact_rows": [
                    {"artifact": "final_model_metrics.csv", "purpose": "Metrics"},
                    {"artifact": "credit_risk_model.pkl", "purpose": "Model"},
                ],
            }
        )
        html = render_to_string("app/reports.html", context, request=self.request("/reports/"))
        self.assertIn("<h1>Reports built for confident model decisions.</h1>", html)
        self.assertIn("Model and governance reports · Aegis-Credit", html)
        self.assertIn('/report-artifacts/final_model_metrics.csv/', html)
        self.assertIn("Restricted release asset", html)
        self.assertIn("<caption", html)

    def test_branded_error_templates_render_without_view_context(self) -> None:
        request = self.request("/")
        for template_name, expected in (
            ("403.html", "Your account cannot open this area"),
            ("404.html", "We could not find that page"),
            ("500.html", "could not complete the request"),
        ):
            with self.subTest(template_name=template_name):
                html = render_to_string(template_name, {}, request=request)
                self.assertIn(expected, html)
                self.assertIn("Skip to main content", html)
