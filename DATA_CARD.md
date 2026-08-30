# Aegis-Credit Demonstration Data Card

## Dataset and source

File: `data/credit_risk.csv`<br>
Source: deterministic generator, `src/generate_demo_data.py`<br>
License: [CC0 1.0](data/LICENSE.md)

The file contains synthetic applicant-like records and simulated binary
outcomes. It is recreated from a fixed seed, contains no real borrowers, and
does not derive from a client or third-party lending dataset.

## Variables

- applicant-like inputs: age, income, employment length, home ownership;
- loan-like inputs: requested amount, intent, synthetic rate and grade;
- credit-like inputs: synthetic previous-default flag and history length;
- simulated target: `loan_status`.

`loan_percent_income` is intentionally serialized at two decimal places and
then recomputed from income and requested amount by the versioned feature
contract. This demonstrates that serving does not trust a client-supplied
derived ratio.

## Limitations

The outcome probability is generated from a simple synthetic relationship to
selected inputs. Any discrimination, calibration, ROC-AUC, threshold, drift,
or fairness figure calculated from it measures only that simulation. It is not
evidence of expected lending performance and is unsuitable for commercial or
operational risk decisions.

Age is excluded from scoring and retained only for form plausibility checks and
limited demonstration monitoring. Grade and interest rate are present solely to
exercise data-quality controls; they are excluded from the score because real
lenders may assign them after underwriting begins.

## Production replacement requirements

Before a controlled pilot, replace this data with a client-approved source and
document source, license, extraction date, product/geography, currency, field
owners, selection effects, target event and observation window, privacy review,
and exact digest. Complete the controls in
[`docs/GOVERNANCE_CHECKLIST.md`](docs/GOVERNANCE_CHECKLIST.md) and independently
validate the resulting model before enabling operational scoring.
