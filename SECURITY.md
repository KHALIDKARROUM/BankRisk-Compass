# Security Policy

## Supported version

Security fixes are applied to the current `main` branch.

## Reporting

Do not publish secrets, applicant data, or exploitable details in a public issue.
Report the problem privately to the repository owner with reproduction steps and
the affected version.

## Operational guidance

- Set a long random `SECRET_KEY` whenever `DEBUG=False`.
- Serve the deployed application only through HTTPS.
- Restrict dashboard access before using non-demo data.
- Set `LOGIN_REQUIRED=True` for every shared deployment and assign least-privilege roles.
- Use PostgreSQL for shared deployments; local SQLite files are not durable service storage.
- Never commit applicant records, credentials, or environment files.
- Keep `AUDIT_HMAC_KEY` separate from public API credentials and rotate it under change control.
- Configure retention and run `purge_old_cases` under an approved schedule.
- Back up and restore-test the case database.
- Treat pickle/joblib model artifacts as trusted-code artifacts. Load only files
  produced by the controlled training pipeline and verify `models/model_manifest.json`.
- CI blocks known advisories with `pip-audit` against the three committed
  requirements files; Dependabot opens weekly dependency and workflow updates.
  Review and merge those updates promptly after the test suite passes.

This repository is a demonstration system and has not undergone an external
penetration test.
