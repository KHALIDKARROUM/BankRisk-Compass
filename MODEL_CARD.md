# Aegis-Credit Model Card

## Identity and status

- Artifact: `2.2.0` unsigned synthetic demonstration model
- Stage: local UI and integration demonstration only
- Target: simulated `loan_status` (`1` = simulated default)
- Output: a model score, risk band, and staff-review recommendation
- Production eligibility: **none**

The checked-in bundle is rebuilt from the same canonical feature contract used
by the web form, API, batch processor, monitoring, and external validation.
The runtime refuses an artifact whose feature-contract version or ordered
feature list differs from the serving contract.

## Intended use

This artifact is for demonstrating an underwriting-workflow interface to
regional and specialty lenders that need a controlled review queue ahead of a
future pilot. It can illustrate input validation, staff handoff, API
idempotency, audit records, and model-governance surfaces.

It must not autonomously approve or decline credit, create adverse-action
reasons, replace identity/affordability/fraud/policy checks, or be used to
estimate lending risk. Its synthetic labels make all reported metrics
demonstration-only.

## Feature contract

The score uses annual income, employment length, home ownership, requested
amount, loan intent, credit-history length, a prior-default flag, and the
canonically derived loan-to-income ratio. The ratio is always reconstructed
from raw income and requested amount; a supplied ratio is overwritten.

Age is excluded from scoring and used only for plausibility validation and
limited demonstration monitoring. Loan grade and interest rate are excluded
because real lenders can assign them after underwriting begins.

## Evaluation evidence

The current report files were regenerated alongside the `2.2.0` demo bundle.
Their ROC-AUC, calibration, fairness, and threshold figures measure only the
synthetic generator's simulated pattern. They are intentionally not published
here as claims about real applicants or prospective clients.

## Required production controls

Before a controlled pilot, a client must supply approved representative data
with mature outcomes. A signed release then requires documented provenance,
external/out-of-time validation, an independent model review, approved
expected-loss assumptions, fair-lending and legal review, a validated
reason-code mapping, monitored drift/calibration/performance, and a tested
rollback/incident/backup process.
