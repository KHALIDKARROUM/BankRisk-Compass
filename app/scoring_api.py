"""Thin HTTP boundary for a loan-origination system scoring request.

Keeping this endpoint outside the browser-focused view module makes the
machine-to-machine contract easier to evolve and test independently.
"""

from __future__ import annotations

import json
import logging
import uuid

from django.conf import settings
from django.http import HttpRequest, JsonResponse
from django.utils.crypto import constant_time_compare

from . import services
from .forms import ApplicantAssessmentForm
from .workflows import IdempotencyConflict


LOGGER = logging.getLogger(__name__)


def score(request: HttpRequest) -> JsonResponse:
    """Validate, score, and persist an idempotent LOS-originated request."""
    configured_keys = dict(getattr(settings, "SCORING_API_KEYS", {}))
    if settings.SCORING_API_KEY:
        configured_keys.setdefault("legacy", settings.SCORING_API_KEY)
    if not configured_keys:
        return JsonResponse({"error": "Scoring API is not configured."}, status=503)

    supplied_key = request.headers.get("X-API-Key", "")
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        supplied_key = authorization.removeprefix("Bearer ").strip()
    client_id = next(
        (
            candidate_id
            for candidate_id, candidate_key in configured_keys.items()
            if constant_time_compare(supplied_key, candidate_key)
        ),
        None,
    )
    remote_address = request.META.get("REMOTE_ADDR", "")
    if client_id is None:
        if services.api_rate_limit_exceeded(f"invalid-auth:{remote_address}"):
            response = JsonResponse({"error": "Rate limit exceeded."}, status=429)
            response["Retry-After"] = "60"
            return response
        return JsonResponse({"error": "Unauthorized."}, status=401)
    identifier = f"client:{client_id}:{remote_address}"
    if services.api_rate_limit_exceeded(identifier):
        response = JsonResponse(
            {"error": "Rate limit exceeded. Try again in one minute."},
            status=429,
        )
        response["Retry-After"] = "60"
        return response

    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Request body must be valid JSON."}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"error": "Request body must be a JSON object."}, status=400)
    allowed_fields = set(ApplicantAssessmentForm.base_fields) - {"request_id"}
    unknown_fields = sorted(set(payload) - allowed_fields)
    if unknown_fields:
        return JsonResponse(
            {"error": "Unknown request fields.", "fields": unknown_fields},
            status=400,
        )

    request_id_value = request.headers.get("Idempotency-Key")
    if request_id_value:
        try:
            request_id = uuid.UUID(request_id_value)
        except ValueError:
            return JsonResponse(
                {"error": "Idempotency-Key must be a valid UUID."},
                status=400,
            )
    else:
        request_id = uuid.uuid4()

    try:
        bundle = services.load_model_bundle()
    except Exception:
        LOGGER.exception("Scoring API model load failed.")
        return JsonResponse({"error": "Model is unavailable."}, status=503)

    payload["request_id"] = request_id
    form = ApplicantAssessmentForm(payload, bundle=bundle)
    if not form.is_valid():
        return JsonResponse(
            {"error": "Validation failed.", "fields": form.errors.get_json_data()},
            status=400,
        )

    blocks = form.distribution_blocks()
    if blocks:
        return JsonResponse(
            {
                "error": "Application is outside the supported model domain; no score was produced.",
                "reasons": blocks,
            },
            status=422,
        )

    # Import lazily: views owns browser-centric persistence and presentation helpers.
    from .views import _case_payload, _persist_result

    result = services.assessment_result(bundle, form.cleaned_data, explain=False)
    try:
        case, created = _persist_result(
            request,
            result,
            bundle,
            form.distribution_warnings(),
            source="api",
            namespace=f"api:{client_id}",
        )
    except IdempotencyConflict as exc:
        return JsonResponse({"error": str(exc)}, status=409)
    response_payload = _case_payload(case)
    response_payload["api_client_id"] = client_id
    response = JsonResponse(response_payload)
    if not created:
        response["Idempotent-Replay"] = "true"
    return response
