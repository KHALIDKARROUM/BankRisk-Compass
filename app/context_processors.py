from __future__ import annotations

from typing import Any

from django.conf import settings
from django.http import HttpRequest

from .access import user_role


NAVIGATION_ITEMS = (
    {
        "key": "overview",
        "label": "Overview",
        "url_name": "overview",
        "roles": {"reviewer", "admin", "local"},
    },
    {
        "key": "assessment",
        "label": "New assessment",
        "url_name": "assessment",
        "roles": {"analyst", "reviewer", "admin", "local"},
    },
    {
        "key": "cases",
        "label": "Cases",
        "url_name": "case-list",
        "roles": {"analyst", "reviewer", "legal", "admin", "local"},
    },
    {
        "key": "batch",
        "label": "Batch load",
        "url_name": "batch-upload",
        "roles": {"analyst", "reviewer", "admin", "local"},
    },
    {
        "key": "monitoring",
        "label": "Monitoring",
        "url_name": "monitoring",
        "roles": {"reviewer", "legal", "admin", "local"},
    },
    {
        "key": "insights",
        "label": "Insights",
        "url_name": "insights",
        "roles": {"reviewer", "admin", "local"},
    },
    {
        "key": "threshold",
        "label": "Threshold",
        "url_name": "threshold",
        "roles": {"reviewer", "admin", "local"},
    },
    {
        "key": "business",
        "label": "Business policy",
        "url_name": "business-policy",
        "roles": {"reviewer", "admin", "local"},
    },
    {
        "key": "reports",
        "label": "Reports",
        "url_name": "reports",
        "roles": {"reviewer", "admin", "local"},
    },
    {
        "key": "api",
        "label": "API",
        "url_name": "api-docs",
        "roles": {"analyst", "reviewer", "admin", "local"},
    },
)


FOOTER_MESSAGES = {
    "overview": "Portfolio evidence supports governance review and does not approve or decline applications.",
    "assessment": "This result supports a staff review. It does not approve or decline an application.",
    "assessment-legacy": "This result supports a staff review. It does not approve or decline an application.",
    "case-list": "Case records support controlled staff review and must be handled under the retention policy.",
    "case-detail": "The model result and the recorded staff review are separate parts of the case record.",
    "batch-upload": "Validate row warnings and errors before using batch results in a staff workflow.",
    "batch-detail": "Validate row warnings and errors before using batch results in a staff workflow.",
    "monitoring": "Monitoring indicators require an owner, a freshness check, and documented follow-up.",
    "insights": "Model insights describe saved validation evidence, not applicant-level decision reasons.",
    "threshold": "Threshold analysis is exploratory and does not change live scoring.",
    "business-policy": "Scenario results do not change live scoring or replace policy approval.",
    "reports": "Reports are governance evidence, not autonomous credit decisions.",
    "api-docs": "API scores support staff review and do not approve or decline applications.",
    "login": "Authorized staff access only.",
}


def product_shell(request: HttpRequest) -> dict[str, Any]:
    """Supply consistent, role-aware shell context to every rendered page.

    Add ``app.context_processors.product_shell`` to the Django template context
    processors.  The dedicated ``product_shell_ready`` flag lets the base
    template distinguish an intentionally empty navigation from legacy views
    that still provide their own ``pages`` list.
    """

    role = user_role(request.user)
    route_name = (
        request.resolver_match.url_name
        if getattr(request, "resolver_match", None) is not None
        else ""
    )
    if settings.LOCAL_DEMO_MODE:
        workspace_label = "Local demo"
        workspace_tone = "demo"
    elif settings.DEBUG:
        workspace_label = "Development workspace"
        workspace_tone = "development"
    else:
        workspace_label = "Controlled workspace"
        workspace_tone = "controlled"

    return {
        "product_shell_ready": True,
        "product_nav_items": [item for item in NAVIGATION_ITEMS if role in item["roles"]],
        "product_role": role,
        "local_demo_mode": settings.LOCAL_DEMO_MODE,
        "batch_processing_inline": settings.BATCH_PROCESS_INLINE,
        "currency_code": settings.CURRENCY_CODE,
        "workspace_badge_label": workspace_label,
        "workspace_badge_tone": workspace_tone,
        "skip_link_text": "Skip to main content",
        "footer_message": FOOTER_MESSAGES.get(
            route_name,
            "Aegis-Credit provides decision support for controlled staff workflows.",
        ),
    }
