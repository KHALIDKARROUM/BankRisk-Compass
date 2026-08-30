"""PDF exports kept separate from model and data services."""

from __future__ import annotations

import io
from typing import Any

from django.conf import settings
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from . import services

def summary_pdf(dashboard: dict[str, Any]) -> bytes:
    summary = services.report_summary(dashboard)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        figure = Figure(figsize=(8.5, 11))
        axis = figure.subplots()
        axis.axis("off")

        y = 0.96
        axis.text(0.05, y, "Aegis-Credit Report", fontsize=18, fontweight="bold", color="#071942")
        y -= 0.055
        axis.text(0.05, y, f"Generated: {services.display_date()}", fontsize=10, color="#60708d")
        if settings.LOCAL_DEMO_MODE:
            y -= 0.035
            axis.text(
                0.05,
                y,
                "LOCAL DEMONSTRATION - NOT APPROVED FOR LENDING DECISIONS",
                fontsize=10,
                fontweight="bold",
                color="#a12622",
            )
        y -= 0.06

        for title, section in [
            ("Release Status", "release_summary"),
            ("Model Summary", "model_summary"),
            ("Dataset Summary", "dataset_summary"),
            ("Threshold Summary", "threshold_summary"),
        ]:
            axis.text(0.05, y, title, fontsize=13, fontweight="bold", color="#062f6c")
            y -= 0.032
            for row in summary[section]:
                axis.text(0.07, y, f"{row['label']}: {row['value']}", fontsize=10, color="#071942")
                y -= 0.026
            y -= 0.018

        axis.text(0.05, y, "Business Recommendation", fontsize=13, fontweight="bold", color="#062f6c")
        y -= 0.034
        axis.text(
            0.07,
            y,
            summary["business_recommendation"],
            fontsize=10,
            color="#071942",
            wrap=True,
        )
        pdf.savefig(figure, bbox_inches="tight")

    buffer.seek(0)
    return buffer.getvalue()


def api_reference_pdf() -> bytes:
    """Create a compact, human-readable PDF reference for the scoring API."""
    brand = colors.HexColor("#5148E8")
    ink = colors.HexColor("#171A3F")
    muted = colors.HexColor("#636980")
    line = colors.HexColor("#E5E7F0")
    pale = colors.HexColor("#F3F2FF")
    warning = colors.HexColor("#FFF2F2")
    buffer = io.BytesIO()
    stylesheet = getSampleStyleSheet()
    styles = {
        "brand": ParagraphStyle("api_brand", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=ink),
        "title": ParagraphStyle("api_title", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=ink, spaceBefore=5, spaceAfter=6),
        "subtitle": ParagraphStyle("api_subtitle", parent=stylesheet["Normal"], fontSize=10.5, leading=15, textColor=muted, spaceAfter=17),
        "section": ParagraphStyle("api_section", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=ink, spaceBefore=10, spaceAfter=8),
        "body": ParagraphStyle("api_body", parent=stylesheet["Normal"], fontSize=9.2, leading=13.5, textColor=ink),
        "table_label": ParagraphStyle("api_label", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=8.4, leading=11, textColor=ink),
        "table_body": ParagraphStyle("api_table_body", parent=stylesheet["Normal"], fontSize=8.7, leading=12, textColor=ink),
        "table_header": ParagraphStyle("api_table_header", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white),
        "note": ParagraphStyle("api_note", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=12, textColor=colors.HexColor("#9B2525")),
        "footer": ParagraphStyle("api_footer", parent=stylesheet["Normal"], fontSize=7.8, leading=10, textColor=muted),
    }

    def paragraph(value: str, style: str = "body") -> Paragraph:
        return Paragraph(value, styles[style])

    def section(title: str) -> list[Any]:
        return [Paragraph(title, styles["section"]), HRFlowable(width="100%", thickness=0.7, color=line, spaceAfter=7)]

    def detail_table(rows: list[tuple[str, str]], *, first_width: float = 1.72 * inch) -> Table:
        table = Table(
            [[paragraph(label, "table_label"), paragraph(detail, "table_body")] for label, detail in rows],
            colWidths=[first_width, 6.15 * inch - first_width],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.35, line),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FAFAFD")),
                ]
            )
        )
        return table

    def draw_page_chrome(pdf_canvas: Any, page_number: int) -> None:
        pdf_canvas.saveState()
        pdf_canvas.setFillColor(brand)
        pdf_canvas.rect(0, A4[1] - 0.42 * inch, A4[0], 0.42 * inch, fill=1, stroke=0)
        pdf_canvas.setFillColor(colors.white)
        pdf_canvas.setFont("Helvetica-Bold", 8.5)
        pdf_canvas.drawString(0.58 * inch, A4[1] - 0.27 * inch, "Aegis-Credit  /  SCORING API REFERENCE")
        pdf_canvas.setStrokeColor(line)
        pdf_canvas.line(0.58 * inch, 0.48 * inch, A4[0] - 0.58 * inch, 0.48 * inch)
        pdf_canvas.setFillColor(muted)
        pdf_canvas.setFont("Helvetica", 7.5)
        pdf_canvas.drawString(0.58 * inch, 0.31 * inch, "Controlled staff workflow - not an autonomous credit decision")
        pdf_canvas.drawRightString(A4[0] - 0.58 * inch, 0.31 * inch, f"Page {page_number}")
        pdf_canvas.restoreState()

    class ApiReferenceCanvas(canvas.Canvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._saved_page_states: list[dict[str, Any]] = []

        def showPage(self) -> None:
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            for page_number, state in enumerate(self._saved_page_states, start=1):
                self.__dict__.update(state)
                draw_page_chrome(self, page_number)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

    schema = services.openapi_schema()
    info = schema.get("info", {})
    version = str(info.get("version", "1.0.0"))
    document = BaseDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.68 * inch,
        title="Aegis-Credit Scoring API Reference",
        author="Aegis-Credit",
    )
    story: list[Any] = [
        Paragraph("Aegis-Credit", styles["brand"]),
        Paragraph("Scoring API reference", styles["title"]),
        Paragraph("A concise guide for authenticated, rate-limited, and idempotent application screening.", styles["subtitle"]),
    ]
    overview = Table(
        [
            [paragraph("ENDPOINT", "table_header"), paragraph("FORMAT", "table_header"), paragraph("OPENAPI VERSION", "table_header")],
            [paragraph("POST /api/v1/score/", "table_body"), paragraph("application/json", "table_body"), paragraph(version, "table_body")],
        ],
        colWidths=[2.2 * inch, 1.7 * inch, 2.25 * inch],
        hAlign="LEFT",
    )
    overview.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), brand),
                ("BACKGROUND", (0, 1), (-1, 1), pale),
                ("BOX", (0, 0), (-1, -1), 0.75, line),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, line),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([overview, Spacer(1, 14), *section("Getting started")])
    story.append(
        detail_table(
            [
                ("Authentication", "Send <b>X-API-Key</b> with the configured scoring client secret. A Bearer token with the same secret is also accepted."),
                ("Idempotency", "Send a UUID in <b>Idempotency-Key</b>. Reuse it only when retrying the same logical request with identical input."),
                ("Result", "A 200 response includes the case ID, model version, probability, risk category, screening result, next step, threshold, and warnings."),
            ]
        )
    )
    story.extend([Spacer(1, 14), *section("Security note")])
    notice = Table([[paragraph("Never place API keys in URLs, source control, or application logs. Store the returned case ID with the approved calling-system audit record.", "note")]], colWidths=[6.15 * inch])
    notice.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), warning), ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#F0C4C4")), ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 11), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story.append(notice)

    story.extend([NextPageTemplate("api-reference"), PageBreak(), *section("Request contract")])
    story.append(Paragraph("Use the following fields in the JSON request body. Applicant reference is optional; every other field is required.", styles["body"]))
    story.append(Spacer(1, 9))
    request_rows = [
        ("Field", "Accepted value and purpose"),
        ("applicant_reference", "Optional internal reference. Do not send names or government identifiers."),
        ("person_age", "Applicant age in years. Used for plausibility checks and excluded from the model score."),
        ("person_income", "Gross annual income in USD."),
        ("person_emp_length", "Years employed, allowing half-year values."),
        ("person_home_ownership", "MORTGAGE, OTHER, OWN, or RENT."),
        ("loan_amnt", "Requested loan amount in USD."),
        ("loan_intent", "DEBTCONSOLIDATION, EDUCATION, HOMEIMPROVEMENT, MEDICAL, PERSONAL, or VENTURE."),
        ("cb_person_cred_hist_length", "Years of credit history."),
        ("cb_person_default_on_file", "N or Y to indicate a previously recorded default."),
    ]
    request_table = Table(
        [[paragraph(label, "table_header" if row_index == 0 else "table_label"), paragraph(detail, "table_header" if row_index == 0 else "table_body")] for row_index, (label, detail) in enumerate(request_rows)],
        colWidths=[2.15 * inch, 4.0 * inch],
        repeatRows=1,
        hAlign="LEFT",
    )
    request_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), brand),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#FAFAFD")),
                ("GRID", (0, 0), (-1, -1), 0.4, line),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(request_table)

    story.extend([NextPageTemplate("api-reference"), PageBreak(), *section("Response and error contract")])
    story.append(Paragraph("Callers should handle status and response fields instead of matching response-message text.", styles["body"]))
    story.append(Spacer(1, 9))
    response_rows = [
        ("Status", "Caller action"),
        ("200 OK", "Store the case ID and continue the staff workflow. An identical replay returns the original result."),
        ("400 Bad Request", "Correct malformed JSON, idempotency key, or field-validation errors. Inspect the returned fields object when present."),
        ("401 Unauthorized", "Check secret injection without logging the key."),
        ("409 Conflict", "Do not retry under the same idempotency key. Investigate the request identity."),
        ("422 Unprocessable Content", "Route the application to the approved non-model process. No score was produced."),
        ("429 Too Many Requests", "Wait for Retry-After before retrying."),
        ("503 Service Unavailable", "Stop scoring and escalate to the service owner."),
    ]
    response_table = Table(
        [[paragraph(label, "table_header" if row_index == 0 else "table_label"), paragraph(detail, "table_header" if row_index == 0 else "table_body")] for row_index, (label, detail) in enumerate(response_rows)],
        colWidths=[1.75 * inch, 4.4 * inch],
        repeatRows=1,
        hAlign="LEFT",
    )
    response_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), brand),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#FAFAFD")),
                ("GRID", (0, 0), (-1, -1), 0.4, line),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(response_table)
    story.extend([Spacer(1, 15), *section("Operational controls")])
    controls = KeepTogether(
        [
            detail_table(
                [
                    ("Rate limiting", "Respect the configured per-minute limit and use backoff after a 429 response."),
                    ("Audit record", "Keep the original payload with the returned case ID in the calling system's approved audit record."),
                    ("Scoring boundary", "The API does not provide explanations or adverse-action reasons. A trained staff member remains responsible for the decision."),
                ]
            ),
            Spacer(1, 10),
            Paragraph("Machine-readable contract: GET /api/v1/openapi.json", styles["footer"]),
        ]
    )
    story.append(controls)
    document.addPageTemplates(
        [
            PageTemplate(
                id="api-reference",
                frames=[Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="body")],
            )
        ]
    )
    document.build(story, canvasmaker=ApiReferenceCanvas)
    buffer.seek(0)
    return buffer.getvalue()


