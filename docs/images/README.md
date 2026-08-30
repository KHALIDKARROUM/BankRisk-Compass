# Image status

## Current release evidence

![Aegis-Credit local demonstration assessment screen](aegis-credit-assessment-2026-08-30.png)

| Capture | Application / model | Viewport | Data | Status |
|---|---|---:|---|---|
| `aegis-credit-assessment-2026-08-30.png` | Aegis-Credit local demo / synthetic model 2.2.0 | Desktop | Blank synthetic-demo form | Current |

The PNG files previously in this directory show the pre-2.1 interface and are
retained only as design history. They are intentionally not embedded in the
current README.

Current product behavior should be verified from the running application. New
release screenshots should be captured after the authenticated and unauthenticated
workflows pass visual review at desktop and mobile widths.

Before replacing the historical images:

1. Use synthetic applicant references and demo-only values; never capture real
   applicant, username, API-key, or reviewer-note data.
2. Capture the workspace-mode badge and the whole page heading so the operating
   context is unambiguous.
3. Include at least the blank assessment, a validation error, a scored result,
   an unreviewed case, a recorded review, batch warnings, stale/missing
   monitoring states, reports, and a role-restricted error page.
4. Review at approximately 1440 px, 940 px, and 390 px widths, plus 200% browser
   zoom and keyboard-only navigation.
5. Check alternative text, page title, heading order, focus visibility, table
   scrolling, and color-independent status labels before committing images.

Record the application version, model version, capture date, viewport, browser,
and demo-data source in the pull request that updates the images. Treat images
as release documentation: replace rather than silently reusing screenshots from
an older workflow.
