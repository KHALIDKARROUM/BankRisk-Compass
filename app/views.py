from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from datetime import timedelta

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.paginator import Paginator
from django.db import connection, transaction
from django.db.migrations.executor import MigrationExecutor
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import services
from .access import (
    access_required,
    batch_queryset_for_user,
    can_manage_legal_holds,
    case_queryset_for_user,
    record_sensitive_access,
    user_role,
)
from .batch_processing import process_batch
from .forms import (
    ApplicantAssessmentForm,
    BatchUploadForm,
    BusinessEconomicsForm,
    CaseAssignmentForm,
    CaseOutcomeForm,
    CaseReviewForm,
    LegalHoldForm,
)
from .models import (
    AssessmentCase,
    BatchAssessment,
    CaseOutcome,
    CaseReviewEvent,
    LegalHoldEvent,
    MonitoringAcknowledgement,
    MonitoringRun,
    PolicyScenario,
    PolicyScenarioEvent,
    PredictionAudit,
)
from .workflows import IdempotencyConflict, persist_assessment


LOGGER = logging.getLogger(__name__)
PAGES = [
    {"key": "assessment", "label": "New assessment", "url_name": "assessment"},
    {"key": "cases", "label": "Cases", "url_name": "case-list"},
    {"key": "batch", "label": "Batch load", "url_name": "batch-upload"},
    {"key": "overview", "label": "Portfolio overview", "url_name": "overview"},
    {"key": "monitoring", "label": "Monitoring", "url_name": "monitoring"},
    {"key": "business", "label": "Business policy", "url_name": "business-policy"},
    {"key": "insights", "label": "Model insights", "url_name": "insights"},
    {"key": "threshold", "label": "Threshold validation", "url_name": "threshold"},
    {"key": "reports", "label": "Reports", "url_name": "reports"},
    {"key": "api", "label": "API", "url_name": "api-docs"},
]


def base_context(active_page: str) -> tuple[dict[str, object], dict[str, object] | None]:
    context: dict[str, object] = {
        "pages": PAGES,
        "active_page": active_page,
        "local_demo_mode": settings.LOCAL_DEMO_MODE,
        "best_model": "Unavailable",
        "model_meta": {"version": "unavailable", "trained_at": "unavailable"},
        "display_date": services.display_date(),
    }

    try:
        dashboard = services.dashboard_data()
    except (FileNotFoundError, services.ArtifactIntegrityError) as exc:
        context["error_message"] = str(exc)
        return context, None

    context["best_model"] = dashboard["best_model"]
    context["model_meta"] = services.model_metadata(dashboard["bundle"])
    return context, dashboard


def assessment_context() -> tuple[dict[str, object], dict[str, object] | None]:
    context: dict[str, object] = {
        "pages": PAGES,
        "active_page": "assessment",
        "local_demo_mode": settings.LOCAL_DEMO_MODE,
        "display_date": services.display_date(),
    }
    try:
        bundle = services.load_model_bundle()
    except (FileNotFoundError, services.ArtifactIntegrityError) as exc:
        # A disabled model is an expected, safe state for a local checkout that
        # contains demonstration data rather than an approved model release.
        LOGGER.warning("Assessment service is unavailable: %s", exc)
        if isinstance(exc, services.ArtifactIntegrityError) and not settings.DATA_PROVENANCE_VERIFIED:
            context.update(
                {
                    "error_title": "Scoring is disabled for this demonstration project",
                    "error_message": (
                        "The bundled dataset and model have not been approved for operational use, "
                        "so Aegis-Credit will not produce loan-risk scores."
                    ),
                    "error_guidance": (
                        "This is expected in a local checkout. Scoring can only be enabled with an "
                        "approved, signed model release and verified data provenance."
                    ),
                }
            )
        else:
            context.update(
                {
                    "error_title": "Assessment service unavailable",
                    "error_message": "The approved model release could not be loaded.",
                    "error_guidance": "Check the model release configuration and try again.",
                }
            )
        return context, None
    return context, bundle


def _actor(request: HttpRequest):
    return request.user if request.user.is_authenticated else None


def _case_payload(case: AssessmentCase) -> dict[str, object]:
    return {
        "case_id": str(case.id),
        "model_version": case.model_version,
        "probability": round(case.probability, 6),
        "risk_category": case.risk_category,
        "screening_result": case.screening_result,
        "recommended_next_step": case.recommendation,
        "threshold": case.threshold,
        "warnings": case.warnings,
        "deployment_stage": case.deployment_stage,
    }


def _persist_result(
    request: HttpRequest,
    result: dict[str, object],
    bundle: dict[str, object],
    warnings: list[str],
    *,
    source: str,
    namespace: str | None = None,
    batch: BatchAssessment | None = None,
) -> tuple[AssessmentCase, bool]:
    if namespace is None:
        actor = _actor(request)
        actor_key = f"user:{actor.pk}" if actor is not None else "local"
        namespace = f"{source}:{actor_key}"
    case, created = persist_assessment(
        actor=_actor(request),
        result=result,
        bundle=bundle,
        warnings=warnings,
        source=source,
        namespace=namespace,
        batch=batch,
    )
    record_sensitive_access(request, "case_created" if created else "case_replayed", case)
    return case, created


def _stored_result_context(case: AssessmentCase, result: dict[str, object]) -> dict[str, object]:
    """Render the stored decision on an idempotent replay, even after a model change."""
    output = dict(result)
    category_class = case.risk_category.lower()
    category_color = {"low": "#07856a", "medium": "#f0a500", "high": "#e21f2d"}.get(
        category_class, "#52627a"
    )
    output.update(
        {
            "probability": case.probability,
            "probability_percent": f"{case.probability:.0%}",
            "gauge_style": f"--score:{case.probability * 100:.1f}%;--color:{category_color};",
            "category": case.risk_category,
            "category_display": f"{case.risk_category} risk",
            "category_class": category_class,
            "category_color": category_color,
            "prediction": case.screening_result,
            "decision": case.recommendation,
            "threshold": case.threshold,
            "explanation_rows": case.explanation_rows,
            "explanation_method": case.explanation_method,
        }
    )
    return output


@access_required("reviewer")
def overview(request: HttpRequest) -> HttpResponse:
    context, dashboard = base_context("overview")
    if dashboard is None:
        return render(request, "app/overview.html", context)

    data = dashboard["data"]
    comparison = dashboard["comparison"]
    importance = dashboard["importance"]
    default_metrics = dashboard["default_metrics"]
    business_metrics = dashboard["business_metrics"]
    threshold = float(dashboard["threshold"])
    default_rate = float(data["loan_status"].mean())

    context.update(
        {
            "metric_tiles": [
                {
                    "label": "Applicants",
                    "value": f"{len(data):,}",
                    "foot": "Rows in data/credit_risk.csv",
                },
                {
                    "label": "Observed Default Rate",
                    "value": services.format_percent(default_rate),
                    "foot": "Target class share",
                },
                {
                    "label": "Best Model",
                    "value": str(dashboard["best_model"]),
                    "foot": f"F1-score {services.format_score(float(dashboard['best_f1']))}",
                },
                {
                    "label": "Business Threshold",
                    "value": f"{threshold:.2f}",
                    "foot": f"Recall {services.format_score(float(business_metrics['recall']))}",
                },
            ],
            "grade_bars": services.default_rate_by_grade(data),
            "intent_bars": services.default_rate_by_intent(data),
            "default_metrics_pairs": services.metric_pairs(
                default_metrics,
                [
                    ("accuracy", "Accuracy", "percent"),
                    ("f1_score", "F1-score", "score"),
                    ("recall", "Recall for defaults", "score"),
                    ("roc_auc", "ROC-AUC", "score"),
                ],
            ),
            "business_metrics_pairs": services.metric_pairs(
                business_metrics,
                [
                    ("accuracy", "Accuracy", "percent"),
                    ("f1_score", "F1-score", "score"),
                    ("recall", "Recall for defaults", "score"),
                ],
            ),
            "business_threshold": f"{float(business_metrics['decision_threshold']):.2f}",
            "model_comparison_table": services.dataframe_table(comparison, digits=3),
            "model_comparison_bars": services.model_comparison_bars(comparison),
            "top_feature": (
                services.pretty_feature_name(str(importance.iloc[0]["feature"]))
                if not importance.empty
                else "Interest Rate"
            ),
            "portfolio_default_rate": services.format_percent(default_rate),
        }
    )
    return render(request, "app/overview.html", context)


@access_required("analyst")
def assessment(request: HttpRequest) -> HttpResponse:
    context, bundle = assessment_context()
    if bundle is None:
        return render(request, "app/assessment.html", context)

    form = ApplicantAssessmentForm(
        request.POST or None,
        bundle=bundle,
        use_demo=request.method == "GET" and request.GET.get("demo") == "1",
    )
    context.update(
        {
            "assessment_form": form,
            "has_result": False,
            "distribution_warnings": [],
        }
    )

    if request.method == "POST" and form.is_valid():
        blocks = form.distribution_blocks()
        if blocks:
            for block in blocks:
                form.add_error(None, block)
            context["no_score_reasons"] = blocks
            return render(request, "app/assessment.html", context)
        result = services.assessment_result(bundle, form.cleaned_data)
        warnings = form.distribution_warnings()
        context["distribution_warnings"] = warnings
        try:
            case, created = _persist_result(
                request,
                result,
                bundle,
                warnings,
                source="web",
            )
        except IdempotencyConflict as exc:
            form.add_error(None, str(exc))
        else:
            context.update(result if created else _stored_result_context(case, result))
            context["has_result"] = True
            context["saved_case"] = case
            context["idempotent_replay"] = not created

    return render(request, "app/assessment.html", context)


@access_required("reviewer")
def insights(request: HttpRequest) -> HttpResponse:
    context, dashboard = base_context("insights")
    if dashboard is None:
        return render(request, "app/insights.html", context)

    context.update(
        {
            "model_comparison_table": services.dataframe_table(
                dashboard["comparison"],
                digits=3,
            ),
            "model_comparison_bars": services.model_comparison_bars(dashboard["comparison"]),
            "importance_bars": services.importance_bars(dashboard["importance"]),
        }
    )
    return render(request, "app/insights.html", context)


@access_required("reviewer")
def threshold_analysis(request: HttpRequest) -> HttpResponse:
    context, dashboard = base_context("threshold")
    if dashboard is None:
        return render(request, "app/threshold.html", context)

    threshold_table = dashboard["threshold_table"]
    default_threshold = float(dashboard["threshold"])
    try:
        selected_threshold = float(request.GET.get("threshold", default_threshold))
    except (TypeError, ValueError):
        selected_threshold = default_threshold
    selected_threshold = min(max(selected_threshold, 0.10), 0.90)
    threshold_context = services.threshold_summary_context(
        threshold_table,
        selected_threshold,
        default_threshold,
    )
    row = threshold_context["row"]

    context.update(
        {
            "selected_threshold": threshold_context["selected_threshold"],
            "business_threshold": threshold_context["business_threshold"],
            "current_threshold_label": threshold_context["current_threshold_label"],
            "business_threshold_label": threshold_context["business_threshold_label"],
            "threshold_summary": [
                {
                    "label": "Current Selected Threshold",
                    "value": f"{float(row.get('threshold', selected_threshold)):.2f}",
                    "foot": "Interactive scenario shown below",
                },
                {
                    "label": "Recommended Business Threshold",
                    "value": f"{default_threshold:.2f}",
                    "foot": "Chosen by 5:1 FN to FP cost",
                },
                {
                    "label": "False Positives",
                    "value": f"{int(row.get('false_positives', 0)):,}",
                    "foot": "Safer applicants routed to review",
                },
                {
                    "label": "False Negatives",
                    "value": f"{int(row.get('false_negatives', 0)):,}",
                    "foot": "Defaults missed by policy",
                },
                {
                    "label": "Business Cost",
                    "value": f"{int(row.get('business_cost', 0)):,}",
                    "foot": "5x FN + 1x FP",
                },
            ],
            "confusion": [
                {
                    "actual": "Actual non-default",
                    "predicted_non_default": f"{int(row.get('true_negatives', 0)):,}",
                    "predicted_default": f"{int(row.get('false_positives', 0)):,}",
                },
                {
                    "actual": "Actual default",
                    "predicted_non_default": f"{int(row.get('false_negatives', 0)):,}",
                    "predicted_default": f"{int(row.get('true_positives', 0)):,}",
                },
            ],
            "threshold_table": services.dataframe_table(threshold_table, digits=3),
        }
    )
    return render(request, "app/threshold.html", context)


@access_required("reviewer")
def reports(request: HttpRequest) -> HttpResponse:
    context, dashboard = base_context("reports")
    if dashboard is None:
        return render(request, "app/reports.html", context)

    summary = services.report_summary(dashboard)
    manifest = json.loads(services.MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    is_demo = settings.LOCAL_DEMO_MODE
    context.update(
        {
            "report_summary": summary,
            "business_report_html": services.markdown_to_html(
                services.load_text_report("business_report.md")
            ),
            "classification_report": services.load_text_report("classification_report.txt")
            or "Run training to generate classification report.",
            "final_metrics_table": services.dataframe_table(
                dashboard["final_metrics"],
                digits=3,
            ),
            "calibration_table": services.dataframe_table(
                dashboard["calibration"],
                digits=3,
            ),
            "fairness_table": services.dataframe_table(
                dashboard["fairness"],
                digits=3,
            ),
            "artifact_rows": [
                {"artifact": "business_report.md", "purpose": "Final written interpretation"},
                {"artifact": "model_comparison.csv", "purpose": "Model comparison metrics"},
                {"artifact": "final_model_metrics.csv", "purpose": "Default and business threshold metrics"},
                {"artifact": "threshold_analysis.csv", "purpose": "Validation-only threshold scenarios"},
                {
                    "artifact": "risk_band_validation.csv",
                    "purpose": "Risk-band monotonicity and uncertainty evidence",
                    "available": not is_demo
                    or (services.REPORTS_DIR / "risk_band_validation.csv").exists(),
                },
                {"artifact": "permutation_importance.csv", "purpose": "Global model drivers"},
                {"artifact": "calibration_analysis.csv", "purpose": "Probability calibration diagnostics"},
                {"artifact": "fairness_age_groups.csv", "purpose": "Age-group monitoring diagnostics"},
                {"artifact": "metric_confidence_intervals.csv", "purpose": "Bootstrap uncertainty intervals"},
            ],
            "report_release_status": (
                "Unapproved demonstration evidence" if is_demo else "Signed approved release evidence"
            ),
            "report_release_tone": "warning" if is_demo else "success",
            "report_release_message": (
                "These mutable demo reports are not an operational release."
                if is_demo
                else "Every downloadable report is hash-bound to the active signed model release."
            ),
            "report_provenance": [
                {"label": "Model version", "value": manifest.get("model_version", "unavailable")},
                {"label": "Generated", "value": manifest.get("trained_at_utc", "unavailable")},
                {"label": "Model digest", "value": manifest.get("model_sha256", "unavailable")},
                {"label": "Dataset digest", "value": manifest.get("data_sha256", "unavailable")},
                {
                    "label": "Policy scenario source",
                    "value": "Threshold validation partition (final test remains locked)",
                },
                {
                    "label": "Approval state",
                    "value": "Demonstration only" if is_demo else "Manifest-attested approved release",
                },
            ],
        }
    )
    return render(request, "app/reports.html", context)


@access_required("reviewer")
def download_summary_csv(request: HttpRequest) -> HttpResponse:
    _, dashboard = base_context("reports")
    if dashboard is None:
        raise Http404("Report summary is unavailable.")

    response = HttpResponse(services.summary_csv(dashboard), content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="aegis_credit_summary.csv"'
    return response


@access_required("reviewer")
def download_summary_pdf(request: HttpRequest) -> HttpResponse:
    _, dashboard = base_context("reports")
    if dashboard is None:
        raise Http404("Report summary is unavailable.")

    response = HttpResponse(services.summary_pdf(dashboard), content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="aegis_credit_summary.pdf"'
    return response


@access_required("reviewer")
def report_artifact(request: HttpRequest, file_name: str) -> FileResponse:
    path = services.report_artifact_path(file_name)
    if path is None:
        raise Http404("Report artifact not found.")

    content_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(path.open("rb"), content_type=content_type or "application/octet-stream")


@access_required("case")
def case_list(request: HttpRequest) -> HttpResponse:
    cases = case_queryset_for_user(request.user).select_related(
        "created_by", "assigned_to", "reviewed_by"
    )
    query = request.GET.get("q", "").strip()
    status = request.GET.get("status", "").strip()
    risk = request.GET.get("risk", "").strip()
    assignee = request.GET.get("assignee", "").strip()
    sla = request.GET.get("sla", "").strip()
    sort = request.GET.get("sort", "newest").strip()
    if query:
        identifier_filter = Q()
        try:
            identifier_filter = Q(id=uuid.UUID(query))
        except ValueError:
            pass
        reference_filter = Q(applicant_reference_digest__in=services.reference_digests(query))
        cases = cases.filter(reference_filter | identifier_filter)
    if status in AssessmentCase.Status.values:
        cases = cases.filter(status=status)
    if risk in {"Low", "Medium", "High"}:
        cases = cases.filter(risk_category__iexact=risk)
    if assignee == "mine" and request.user.is_authenticated:
        cases = cases.filter(assigned_to=request.user)
    elif assignee == "unassigned":
        cases = cases.filter(assigned_to=None)
    elif assignee:
        try:
            cases = cases.filter(assigned_to_id=int(assignee))
        except ValueError:
            assignee = ""

    now = timezone.now()
    open_filter = ~Q(status=AssessmentCase.Status.CLOSED)
    if sla == "overdue":
        cases = cases.filter(open_filter, due_at__lt=now)
    elif sla == "due_soon":
        cases = cases.filter(open_filter, due_at__gte=now, due_at__lte=now + timedelta(hours=24))
    elif sla == "no_due_date":
        cases = cases.filter(due_at=None)

    sort_options = {
        "newest": "-created_at",
        "oldest": "created_at",
        "highest_risk": "-probability",
        "due_first": "due_at",
    }
    cases = cases.order_by(sort_options.get(sort, "-created_at"), "-created_at")
    page_obj = Paginator(cases, settings.CASE_PAGE_SIZE).get_page(request.GET.get("page"))
    for case in page_obj.object_list:
        if case.status == AssessmentCase.Status.CLOSED or not case.due_at:
            case.sla_label = "Closed" if case.status == AssessmentCase.Status.CLOSED else "No due date"
            case.sla_tone = "neutral"
        elif case.due_at < now:
            case.sla_label = "Overdue"
            case.sla_tone = "danger"
        elif case.due_at <= now + timedelta(hours=24):
            case.sla_label = "Due soon"
            case.sla_tone = "warning"
        else:
            case.sla_label = "On track"
            case.sla_tone = "success"
        record_sensitive_access(request, "case_listed", case)
    can_see_all_assignees = user_role(request.user) in {"reviewer", "legal", "admin", "local"}
    assignee_choices: list[tuple[str, str]] = [("unassigned", "Unassigned")]
    if request.user.is_authenticated:
        assignee_choices.insert(0, ("mine", "My queue"))
    if can_see_all_assignees:
        reviewers = get_user_model().objects.filter(
            Q(groups__name="Reviewers") | Q(groups__name="Administrators"), is_active=True
        ).distinct()
        assignee_choices.extend((str(user.pk), user.get_username()) for user in reviewers)
    context = {
        "pages": PAGES,
        "active_page": "cases",
        "display_date": services.display_date(),
        "cases": page_obj.object_list,
        "page_obj": page_obj,
        "query": query,
        "selected_status": status,
        "selected_risk": risk,
        "selected_assignee": assignee,
        "selected_sla": sla,
        "selected_sort": sort,
        "sort_choices": [
            ("newest", "Newest first"),
            ("oldest", "Oldest first"),
            ("highest_risk", "Highest score first"),
            ("due_first", "Earliest due first"),
        ],
        "assignee_choices": assignee_choices,
        "sla_choices": [
            ("overdue", "Overdue"),
            ("due_soon", "Due in 24 hours"),
            ("no_due_date", "No due date"),
        ],
        "case_sla_enabled": True,
        "status_choices": AssessmentCase.Status.choices,
        "risk_choices": ["Low", "Medium", "High"],
    }
    return render(request, "app/case_list.html", context)


REVIEW_TRANSITIONS = {
    AssessmentCase.Status.NEW: {AssessmentCase.Status.NEW, AssessmentCase.Status.IN_REVIEW},
    AssessmentCase.Status.IN_REVIEW: {
        AssessmentCase.Status.IN_REVIEW,
        AssessmentCase.Status.REFERRED,
        AssessmentCase.Status.CLEARED,
        AssessmentCase.Status.CLOSED,
    },
    AssessmentCase.Status.REFERRED: {
        AssessmentCase.Status.REFERRED,
        AssessmentCase.Status.IN_REVIEW,
        AssessmentCase.Status.CLEARED,
        AssessmentCase.Status.CLOSED,
    },
    AssessmentCase.Status.CLEARED: {
        AssessmentCase.Status.CLEARED,
        AssessmentCase.Status.IN_REVIEW,
        AssessmentCase.Status.CLOSED,
    },
    AssessmentCase.Status.CLOSED: {AssessmentCase.Status.CLOSED, AssessmentCase.Status.IN_REVIEW},
}


def _review_state(case: AssessmentCase) -> dict[str, object]:
    return {
        "status": case.status,
        "assigned_to_id": case.assigned_to_id,
        "reviewed_by_id": case.reviewed_by_id,
        "override_decision": case.override_decision,
        "override_reason": case.override_reason,
        "reviewer_notes": case.reviewer_notes,
        "legal_hold": case.legal_hold,
        "reviewed_at": case.reviewed_at.isoformat() if case.reviewed_at else None,
    }


@access_required("case")
def case_detail(request: HttpRequest, case_id: uuid.UUID) -> HttpResponse:
    case = get_object_or_404(
        case_queryset_for_user(request.user).select_related(
            "created_by", "assigned_to", "reviewed_by"
        ),
        id=case_id,
    )
    record_sensitive_access(request, "case_viewed", case)
    role = user_role(request.user)
    can_review = role in {"reviewer", "admin", "local"}
    can_assign = role in {"reviewer", "admin", "local"}
    can_record_outcome = role in {"reviewer", "admin", "local"}
    can_legal_hold = can_manage_legal_holds(request.user)
    action = request.POST.get("form_action", "") if request.method == "POST" else ""

    review_form = CaseReviewForm(
        request.POST if action == "review" else None,
        instance=case,
        prefix="review",
    )
    assignment_form = CaseAssignmentForm(
        request.POST if action == "assignment" else None,
        case=case,
        prefix="assignment",
    )
    legal_hold_form = LegalHoldForm(
        request.POST if action == "legal_hold" else None,
        case=case,
        prefix="legal",
    )
    outcome_exists = CaseOutcome.objects.filter(case=case).exists()
    outcome_form = CaseOutcomeForm(
        request.POST if action == "outcome" else None,
        prefix="outcome",
        initial={"as_of_date": timezone.localdate()},
    )

    if request.method == "POST" and action == "review":
        if not can_review:
            return render(request, "403.html", status=403)
        if review_form.is_valid():
            with transaction.atomic():
                locked = AssessmentCase.objects.select_for_update().get(id=case.id)
                if review_form.cleaned_data["expected_version"] != locked.review_version:
                    review_form.add_error(None, "This case changed in another session. Reload and try again.")
                elif review_form.cleaned_data["status"] not in REVIEW_TRANSITIONS[locked.status]:
                    review_form.add_error("status", "That review status transition is not allowed.")
                else:
                    before = _review_state(locked)
                    locked.status = review_form.cleaned_data["status"]
                    locked.reviewer_notes = review_form.cleaned_data["reviewer_notes"]
                    locked.override_decision = review_form.cleaned_data["override_decision"]
                    locked.override_reason = review_form.cleaned_data["override_reason"]
                    locked.reviewed_at = timezone.now()
                    locked.reviewed_by = _actor(request)
                    if locked.assigned_to_id is None and request.user.is_authenticated:
                        locked.assigned_to = request.user
                    locked.review_version += 1
                    locked.save()
                    CaseReviewEvent.objects.create(
                        case=locked,
                        actor=_actor(request),
                        event_type=CaseReviewEvent.EventType.REVIEW,
                        review_version=locked.review_version,
                        before_state=before,
                        after_state=_review_state(locked),
                        reason=locked.override_reason or "Review record updated.",
                    )
                    record_sensitive_access(request, "case_reviewed", locked)
                    messages.success(request, "Review saved as an immutable event.")
                    return redirect("case-detail", case_id=case.id)

    if request.method == "POST" and action == "assignment":
        if not can_assign:
            return render(request, "403.html", status=403)
        if assignment_form.is_valid():
            with transaction.atomic():
                locked = AssessmentCase.objects.select_for_update().get(id=case.id)
                if assignment_form.cleaned_data["expected_version"] != locked.review_version:
                    assignment_form.add_error(None, "This case changed in another session. Reload and try again.")
                else:
                    before = _review_state(locked)
                    locked.assigned_to = assignment_form.cleaned_data["assigned_to"]
                    locked.review_version += 1
                    locked.save(update_fields=["assigned_to", "review_version", "updated_at"])
                    CaseReviewEvent.objects.create(
                        case=locked,
                        actor=_actor(request),
                        event_type=CaseReviewEvent.EventType.ASSIGNMENT,
                        review_version=locked.review_version,
                        before_state=before,
                        after_state=_review_state(locked),
                        reason="Case assignment changed.",
                    )
                    messages.success(request, "Assignment saved.")
                    return redirect("case-detail", case_id=case.id)

    if request.method == "POST" and action == "legal_hold":
        if not can_legal_hold:
            return render(request, "403.html", status=403)
        if legal_hold_form.is_valid():
            with transaction.atomic():
                locked = AssessmentCase.objects.select_for_update().get(id=case.id)
                if legal_hold_form.cleaned_data["expected_version"] != locked.review_version:
                    legal_hold_form.add_error(None, "This case changed in another session. Reload and try again.")
                else:
                    hold_action = legal_hold_form.cleaned_data["action"]
                    locked.legal_hold = hold_action == LegalHoldEvent.Action.PLACED
                    locked.review_version += 1
                    locked.save(update_fields=["legal_hold", "review_version", "updated_at"])
                    LegalHoldEvent.objects.create(
                        case=locked,
                        actor=_actor(request),
                        action=hold_action,
                        reason=legal_hold_form.cleaned_data["reason"],
                        ticket_reference=legal_hold_form.cleaned_data["ticket_reference"],
                    )
                    record_sensitive_access(request, f"legal_hold_{hold_action}", locked)
                    messages.success(request, f"Legal hold {hold_action} with an immutable record.")
                    return redirect("case-detail", case_id=case.id)

    if request.method == "POST" and action == "outcome":
        if not can_record_outcome:
            return render(request, "403.html", status=403)
        if outcome_exists:
            outcome_form.add_error(None, "A mature outcome is already recorded for this case.")
        elif outcome_form.is_valid():
            with transaction.atomic():
                locked = AssessmentCase.objects.select_for_update().get(id=case.id)
                if CaseOutcome.objects.filter(case=locked).exists():
                    outcome_form.add_error(None, "A mature outcome was recorded in another session.")
                else:
                    outcome = outcome_form.save(commit=False)
                    outcome.case = locked
                    outcome.recorded_by = _actor(request)
                    outcome.save()
                    before = _review_state(locked)
                    locked.review_version += 1
                    locked.save(update_fields=["review_version", "updated_at"])
                    CaseReviewEvent.objects.create(
                        case=locked,
                        actor=_actor(request),
                        event_type=CaseReviewEvent.EventType.OUTCOME,
                        review_version=locked.review_version,
                        before_state=before,
                        after_state={**_review_state(locked), "outcome": outcome.outcome},
                        reason="Mature performance outcome recorded.",
                    )
                    record_sensitive_access(request, "case_outcome_recorded", locked)
                    messages.success(request, "Mature outcome recorded and locked.")
                    return redirect("case-detail", case_id=case.id)

    application_rows = [
        {
            "feature": services.pretty_feature_name(feature),
            "value": (
                services.format_money(float(value))
                if feature in {"person_income", "loan_amnt"}
                else services.format_percent(float(value))
                if feature == "loan_percent_income"
                else str(value)
            ),
        }
        for feature, value in case.application_data.items()
    ]
    context = {
        "pages": PAGES,
        "active_page": "cases",
        "display_date": services.display_date(),
        "case": case,
        "review_form": review_form,
        "assignment_form": assignment_form,
        "legal_hold_form": legal_hold_form,
        "outcome_form": outcome_form,
        "outcome": CaseOutcome.objects.filter(case=case).first(),
        "review_events": case.review_events.select_related("actor")[:50],
        "legal_hold_events": case.legal_hold_events.select_related("actor")[:50],
        "can_review": can_review,
        "can_assign": can_assign,
        "can_legal_hold": can_legal_hold,
        "can_record_outcome": can_record_outcome and not outcome_exists,
        "application_rows": application_rows,
    }
    return render(request, "app/case_detail.html", context)


@access_required("analyst")
def batch_upload(request: HttpRequest) -> HttpResponse:
    context, bundle = assessment_context()
    context["active_page"] = "batch"
    if bundle is None:
        return render(request, "app/batch_upload.html", context)

    form = BatchUploadForm(request.POST or None, request.FILES or None)
    context["batch_form"] = form
    recent_batches = list(batch_queryset_for_user(request.user)[:20])
    for recent_batch in recent_batches:
        record_sensitive_access(request, "batch_listed", recent_batch)
    context["recent_batches"] = recent_batches
    if request.method == "POST" and form.is_valid():
        upload = form.cleaned_data["file"]
        upload.seek(0)
        upload_payload = upload.read()
        batch = BatchAssessment.objects.create(
            created_by=_actor(request),
            file_name=upload.name,
            upload_payload=upload_payload,
            status=BatchAssessment.Status.PENDING,
        )
        record_sensitive_access(request, "batch_created", batch)
        if settings.BATCH_PROCESS_INLINE:
            try:
                process_batch(batch.id)
            except Exception as exc:
                LOGGER.exception("Batch %s failed", batch.id)
                messages.error(
                    request,
                    "The batch was saved but processing failed. Review the protected batch details or retry.",
                )
            else:
                batch.refresh_from_db()
                messages.success(
                    request,
                    f"Processed {batch.total_rows:,} rows: {batch.valid_rows:,} scored and "
                    f"{batch.invalid_rows:,} invalid.",
                )
        else:
            messages.success(
                request,
                "Batch queued. Its progress and row-level results are durable and retryable.",
            )
        return redirect("batch-detail", batch_id=batch.id)

    return render(request, "app/batch_upload.html", context)


@access_required("analyst")
def batch_detail(request: HttpRequest, batch_id: uuid.UUID) -> HttpResponse:
    batch = get_object_or_404(batch_queryset_for_user(request.user).prefetch_related("rows"), id=batch_id)
    record_sensitive_access(request, "batch_viewed", batch)
    rows = list(batch.rows.select_related("case").all())
    warning_count = sum(bool(row.warnings) for row in rows)
    processed_rows = sum(row.status != "pending" for row in rows)
    progress = (
        round(processed_rows / batch.total_rows * 100)
        if batch.total_rows
        else None
    )
    return render(
        request,
        "app/batch_detail.html",
        {
            "pages": PAGES,
            "active_page": "batch",
            "display_date": services.display_date(),
            "batch": batch,
            "batch_rows": rows,
            "batch_warning_count": warning_count,
            "batch_progress_percent": progress,
            "batch_error_message": batch.error_message,
            "batch_can_download": bool(rows),
        },
    )


@require_POST
@access_required("analyst")
def batch_cancel(request: HttpRequest, batch_id: uuid.UUID) -> HttpResponse:
    batch = get_object_or_404(batch_queryset_for_user(request.user), id=batch_id)
    if batch.status not in {BatchAssessment.Status.PENDING, BatchAssessment.Status.PROCESSING}:
        messages.error(request, "Only pending or processing batches can be cancelled.")
    else:
        batch.cancel_requested = True
        batch.save(update_fields=["cancel_requested"])
        record_sensitive_access(request, "batch_cancel_requested", batch)
        messages.success(request, "Cancellation requested.")
    return redirect("batch-detail", batch_id=batch.id)


@require_POST
@access_required("analyst")
def batch_retry(request: HttpRequest, batch_id: uuid.UUID) -> HttpResponse:
    batch = get_object_or_404(batch_queryset_for_user(request.user), id=batch_id)
    if batch.status != BatchAssessment.Status.FAILED:
        messages.error(request, "Only failed batches can be retried.")
        return redirect("batch-detail", batch_id=batch.id)
    batch.status = BatchAssessment.Status.PENDING
    batch.cancel_requested = False
    batch.save(update_fields=["status", "cancel_requested"])
    record_sensitive_access(request, "batch_retry_queued", batch)
    if settings.BATCH_PROCESS_INLINE:
        try:
            process_batch(batch.id)
        except Exception:
            LOGGER.exception("Retried batch %s failed", batch.id)
            messages.error(request, "The retry failed; row-level progress was retained.")
    return redirect("batch-detail", batch_id=batch.id)


@access_required("analyst")
def batch_template(request: HttpRequest) -> HttpResponse:
    response = HttpResponse(services.batch_template_csv(), content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="aegis_credit_batch_template.csv"'
    return response


@access_required("analyst")
def batch_results(request: HttpRequest, batch_id: uuid.UUID) -> HttpResponse:
    batch = get_object_or_404(batch_queryset_for_user(request.user), id=batch_id)
    record_sensitive_access(request, "batch_results_exported", batch)
    response = HttpResponse(
        services.batch_results_csv(batch.results),
        content_type="text/csv",
    )
    response["Content-Disposition"] = (
        f'attachment; filename="aegis_credit_batch_{batch.id}_results.csv"'
    )
    return response


@access_required("monitoring")
def monitoring(request: HttpRequest) -> HttpResponse:
    audits = PredictionAudit.objects.all()
    cases = AssessmentCase.objects.all()
    selected_model = request.GET.get("model_version", "").strip()
    if selected_model:
        audits = audits.filter(model_version=selected_model)
        cases = cases.filter(model_version=selected_model)
    daily = list(
        audits.annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(
            scores=Count("id"),
            average_probability=Avg("probability"),
        )
        .order_by("-day")[:30]
    )
    by_risk = list(
        audits.values("risk_category")
        .annotate(count=Count("id"), average_probability=Avg("probability"))
        .order_by("-count")
    )
    by_source = list(audits.values("source").annotate(count=Count("id")).order_by("-count"))
    reviewed = cases.exclude(reviewed_at=None).count()
    overridden = cases.exclude(override_decision="").count()
    latest_run = MonitoringRun.objects.filter(
        model_version=selected_model
    ).first() if selected_model else MonitoringRun.objects.first()
    if latest_run:
        drift_rows = latest_run.metrics.get("drift_rows", [])
        drift = __import__("pandas").DataFrame(drift_rows)
        freshness_hours = getattr(settings, "MONITORING_FRESHNESS_HOURS", 24)
        stale = latest_run.created_at < timezone.now() - timedelta(hours=freshness_hours)
        monitoring_status = "stale" if stale else "fresh"
        monitoring_message = (
            f"Owned by {latest_run.owner}; window {latest_run.window_start} to "
            f"{latest_run.window_end}."
        )
    else:
        drift = __import__("pandas").DataFrame()
        monitoring_status = "missing"
        monitoring_message = "No traceable monitoring run has been recorded."
    outcomes = CaseOutcome.objects.select_related("case")
    if selected_model:
        outcomes = outcomes.filter(case__model_version=selected_model)
    performance = services.outcome_performance_table(outcomes)
    context = {
        "pages": PAGES,
        "active_page": "monitoring",
        "display_date": services.display_date(),
        "total_scores": audits.count(),
        "open_cases": cases.exclude(status=AssessmentCase.Status.CLOSED).count(),
        "reviewed_cases": reviewed,
        "override_rate": f"{(overridden / reviewed if reviewed else 0):.1%}",
        "daily": daily,
        "by_risk": by_risk,
        "by_source": by_source,
        "drift_table": services.dataframe_table(drift, digits=4),
        "performance_table": services.dataframe_table(performance, digits=4),
        "mature_outcomes": outcomes.exclude(
            outcome=CaseOutcome.Outcome.CLOSED_OTHER
        ).count(),
        "latest_monitoring_run": latest_run,
        "monitoring_status": monitoring_status,
        "monitoring_message": monitoring_message,
        "drift_generated_at": latest_run.created_at if latest_run else None,
        "drift_model_version": latest_run.model_version if latest_run else None,
        "drift_sample_size": latest_run.sample_size if latest_run else None,
        "drift_alert_count": len(latest_run.alerts) if latest_run else None,
        "model_versions": PredictionAudit.objects.values_list("model_version", flat=True).distinct(),
        "selected_model": selected_model,
        "monitoring_acknowledgements": (
            latest_run.acknowledgements.select_related("actor")[:20] if latest_run else []
        ),
    }
    return render(request, "app/monitoring.html", context)


@require_POST
@access_required("reviewer")
def monitoring_acknowledge(request: HttpRequest, run_id: uuid.UUID) -> HttpResponse:
    run = get_object_or_404(MonitoringRun, id=run_id)
    action = request.POST.get("action", "").strip()
    note = request.POST.get("note", "").strip()
    if action not in MonitoringAcknowledgement.Action.values:
        messages.error(request, "Choose a valid monitoring action.")
    elif len(note) < 10:
        messages.error(request, "Record a substantive monitoring note.")
    else:
        MonitoringAcknowledgement.objects.create(
            run=run,
            actor=_actor(request),
            action=action,
            note=note,
        )
        messages.success(request, f"Monitoring run {action} with an immutable note.")
    return redirect("monitoring")


@access_required("reviewer")
def business_policy(request: HttpRequest) -> HttpResponse:
    context, dashboard = base_context("business")
    if dashboard is None:
        return render(request, "app/business_policy.html", context)
    submitted = request.POST if request.method == "POST" else request.GET
    form = BusinessEconomicsForm(
        submitted
        or {
            "scenario_name": "",
            "average_exposure": "10000",
            "loss_given_default": "0.60",
            "annual_margin": "0.08",
            "review_cost": "35",
            "review_abandonment_rate": "0.02",
            "downstream_default_catch_rate": "0.50",
            "review_capacity": "1000",
        }
    )
    if form.is_valid():
        economics = services.business_economics(
            dashboard["threshold_table"],
            average_exposure=float(form.cleaned_data["average_exposure"]),
            loss_given_default=float(form.cleaned_data["loss_given_default"]),
            annual_margin=float(form.cleaned_data["annual_margin"]),
            review_cost=float(form.cleaned_data["review_cost"]),
            review_abandonment_rate=float(form.cleaned_data["review_abandonment_rate"]),
            downstream_default_catch_rate=float(
                form.cleaned_data["downstream_default_catch_rate"]
            ),
            review_capacity=int(form.cleaned_data["review_capacity"]),
        )
    else:
        economics = {"table": dashboard["threshold_table"], "recommended": {}}
    recommended = economics.get("recommended", {})
    recommended_display = (
        {
            "threshold": f"{float(recommended['threshold']):.2f}",
            "estimated_total_cost": services.format_money(
                float(recommended["estimated_total_cost"])
            ),
            "review_rate": services.format_percent(float(recommended["review_rate"])),
            "recall": services.format_score(float(recommended["recall"])),
        }
        if recommended
        else {}
    )
    display_columns = [
        "threshold",
        "precision",
        "recall",
        "review_rate",
        "missed_default_loss",
        "manual_review_cost",
        "false_positive_opportunity_cost",
        "estimated_total_cost",
        "within_review_capacity",
    ]
    table = economics.get("table")
    if table is not None and not table.empty and "estimated_total_cost" in table:
        table = table[display_columns]
    if request.method == "POST" and form.is_valid() and request.POST.get("form_action") == "save":
        name = str(form.cleaned_data.get("scenario_name", "")).strip()
        if not name:
            form.add_error("scenario_name", "Give the scenario a name before saving it.")
        else:
            latest_version = (
                PolicyScenario.objects.filter(name=name).order_by("-version").values_list("version", flat=True).first()
                or 0
            )
            scenario = PolicyScenario.objects.create(
                name=name,
                version=latest_version + 1,
                created_by=_actor(request),
                assumptions=economics.get("assumptions", {}),
                results={
                    "recommended": recommended,
                    "illustrative_only": True,
                    "evaluation_source": "threshold_selection_validation",
                },
                model_version=str(dashboard["bundle"].get("model_version", "legacy")),
            )
            PolicyScenarioEvent.objects.create(
                scenario=scenario,
                actor=_actor(request),
                action=PolicyScenarioEvent.Action.CREATED,
                reason="Draft scenario saved for independent review.",
            )
            messages.success(request, f"Saved scenario {name} v{latest_version + 1} as a draft.")
            return redirect("business-policy")
    context.update(
        {
            "economics_form": form,
            "recommended": recommended,
            "recommended_display": recommended_display,
            "economics_table": services.dataframe_table(table, digits=2, max_rows=81),
            "model_threshold": dashboard["threshold"],
            "policy_scenarios": PolicyScenario.objects.select_related(
                "created_by", "approved_by"
            )[:50],
            "scenario_is_illustrative": True,
        }
    )
    return render(request, "app/business_policy.html", context)


@require_POST
@access_required("admin")
def policy_scenario_decision(request: HttpRequest, scenario_id: uuid.UUID) -> HttpResponse:
    # Local demo mode relaxes page authentication for exploration, but an
    # approval is a governance act and must always have an accountable actor.
    if not request.user.is_authenticated:
        return render(request, "403.html", status=403)
    action = request.POST.get("decision", "").strip()
    reason = request.POST.get("reason", "").strip()
    if action not in {PolicyScenario.Status.APPROVED, PolicyScenario.Status.REJECTED}:
        messages.error(request, "Choose approve or reject.")
        return redirect("business-policy")
    if len(reason) < 10:
        messages.error(request, "Record a substantive approval or rejection reason.")
        return redirect("business-policy")
    with transaction.atomic():
        scenario = get_object_or_404(PolicyScenario.objects.select_for_update(), id=scenario_id)
        if scenario.status != PolicyScenario.Status.DRAFT:
            messages.error(request, "Only draft scenarios can receive a decision.")
            return redirect("business-policy")
        scenario.status = action
        if action == PolicyScenario.Status.APPROVED:
            scenario.approved_at = timezone.now()
            scenario.approved_by = _actor(request)
        scenario.save(update_fields=["status", "approved_at", "approved_by", "updated_at"])
        PolicyScenarioEvent.objects.create(
            scenario=scenario,
            actor=_actor(request),
            action=(
                PolicyScenarioEvent.Action.APPROVED
                if action == PolicyScenario.Status.APPROVED
                else PolicyScenarioEvent.Action.REJECTED
            ),
            reason=reason,
        )
    messages.success(request, f"Scenario {scenario.name} v{scenario.version} was {action}.")
    return redirect("business-policy")


@require_GET
@access_required("analyst")
def api_docs(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "app/api_docs.html",
        {
            "pages": PAGES,
            "active_page": "api",
            "display_date": services.display_date(),
            "api_enabled": bool(settings.SCORING_API_KEY or settings.SCORING_API_KEYS),
        },
    )


@require_GET
def openapi_json(request: HttpRequest) -> JsonResponse:
    return JsonResponse(services.openapi_schema())


@require_GET
@access_required("analyst")
def download_api_reference_pdf(request: HttpRequest) -> HttpResponse:
    response = HttpResponse(services.api_reference_pdf(), content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="aegis-credit-scoring-api-reference.pdf"'
    return response


@never_cache
def health(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok"})


@never_cache
def readiness(request: HttpRequest) -> JsonResponse:
    try:
        bundle = services.load_model_bundle()
        services.load_credit_data()
        services.load_report_csv("final_model_metrics.csv")
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        if MigrationExecutor(connection).migration_plan(
            MigrationExecutor(connection).loader.graph.leaf_nodes()
        ):
            raise RuntimeError("Database migrations are pending.")
        cache_key = f"readiness:{uuid.uuid4()}"
        cache.set(cache_key, "ok", timeout=10)
        if cache.get(cache_key) != "ok":
            raise RuntimeError("Cache read/write check failed.")
        cache.delete(cache_key)
    except Exception as exc:
        LOGGER.exception("Readiness check failed: %s", exc)
        return JsonResponse({"status": "not-ready"}, status=503)

    return JsonResponse(
        {
            "status": "ready",
            "model_version": str(bundle.get("model_version", "legacy")),
            "deployment_stage": (
                AssessmentCase.DeploymentStage.LOCAL_DEMO
                if settings.LOCAL_DEMO_MODE
                else AssessmentCase.DeploymentStage.APPROVED
            ),
        }
    )


@csrf_exempt
@require_POST
def score_api(request: HttpRequest) -> JsonResponse:
    from .scoring_api import score

    return score(request)
