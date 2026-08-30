# Loan-Origination System Integration Guide

## Intended deployment

Each Aegis-Credit deployment is dedicated to one lender organization and one
isolated database. It is not a shared multi-tenant service. Create separate
deployments, databases, encryption keys, and API credentials for separate
clients.

## Synchronous score-and-case workflow

The supported integration is a server-to-server call from the loan-origination
system (LOS) to `POST /api/v1/score/`.

1. The LOS creates a UUID for the logical application submission.
2. It sends the application JSON with that UUID as `Idempotency-Key` and its
   client-specific secret as `X-API-Key`.
3. Aegis-Credit validates the request, persists a case and an audit record,
   then returns the score and durable `case_id`.
4. The LOS stores the `case_id`, model version, deployment stage, and original
   idempotency key with its own application record.
5. A timeout is retried with the same idempotency key and identical payload;
   the response is safely replayed. A changed payload under the same key is
   rejected with `409`.

The API is intentionally a screening handoff. A downstream LOS must continue
identity, affordability, fraud, policy, compliance, and all final decision
workflows itself.

## Credential and environment setup

Set `SCORING_API_KEYS` to a JSON map of an LOS client identifier to a unique
secret of at least 32 characters. Rotate a client by introducing a new client
identifier, migrating the caller, then removing the old entry under change
control. Do not send API keys in URLs, browser code, or logs.

Example (secret values are placeholders):

```text
SCORING_API_KEYS={"primary-los":"replace-with-a-unique-32-character-secret"}
```

The API returns the matched `api_client_id` so calling-system logs can reconcile
which configured integration produced a case without recording the secret.

## Reconciliation and failures

- `200`: persist the returned case ID and continue the normal staff workflow.
- `400` / `422`: correct source data or route to the approved non-model path.
- `401`: stop retries and investigate credential injection without logging it.
- `409`: investigate the caller's application identity; do not change the
  existing case silently.
- `429`: retry only after the returned `Retry-After` period.
- `503`: stop scoring and use the lender's approved continuity process.

Daily reconciliation should compare LOS submission UUIDs to Aegis-Credit case
IDs and investigate missing, duplicate, or cross-environment records. Use the
case ID rather than an applicant name or account number as the integration key.

## Pilot acceptance checks

Before connecting a real LOS, test a non-production deployment with synthetic
records: valid scoring, validation errors, timeout/replay, duplicate-key
conflict, rate-limit handling, role-restricted case access, retention, backup
restore, and model-unavailable behavior. Production scoring remains disabled
until a signed, provenance-approved model release is independently validated.
