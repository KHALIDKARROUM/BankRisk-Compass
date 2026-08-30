# ML Development and Release Contract

## Canonical feature construction

`src.feature_contract` is the single source of truth for model input ordering
and derived features. `loan_percent_income` is always reconstructed as
`loan_amnt / person_income`, rounded to four decimal places. A value supplied in
a CSV or request is ignored and overwritten.

Training, external validation, drift monitoring, and application scoring must
all call:

```python
from src.feature_contract import model_feature_frame

application = model_feature_frame(raw_application_frame)
```

Raw application data therefore needs income, employment length, loan amount,
credit-history length, home ownership, loan intent, and the prior-default flag.
Age can be retained outside the model frame for plausibility and approved
fairness monitoring. Grade and interest rate remain excluded from scoring.

Any change to derivation, ordering, units, or category semantics requires a new
`FEATURE_CONTRACT_VERSION`, model version, regenerated evaluation, and signed
release.

## Evaluation partitions

Model fitting, probability calibration, threshold selection, candidate model
selection, and final testing remain separate. The interactive
`threshold_analysis.csv` is generated from the threshold-selection partition.
The final test is evaluated only at the predeclared `0.50` and selected business
thresholds and must not be used to choose a later policy.

## Release rules

A release must:

- run the full training configuration; `--quick --release` is rejected;
- use a clean commit tagged exactly `model-v<MODEL_VERSION>`;
- have independently approved data provenance;
- use the Ed25519 private key supplied by the release secret manager;
- write the model to a content-addressed immutable directory;
- snapshot generated reports beside that model and sign every report digest in
  the manifest.

The checked-in `2.2.0` model is an unsigned synthetic demonstration artifact
and must not be promoted. It is rebuilt with `python -m src.train_model --demo
--quick` after regenerating the deterministic data. Regenerating a signed
model remains blocked until a separate production dataset's provenance
checklist is completed and release credentials are available.

## Monitoring and external validation

Production commands fail closed unless the manifest identifies a valid signed
release and `MODEL_SIGNING_PUBLIC_KEY` is available:

```bash
python -m src.monitor_model --data incoming.csv
python -m src.validate_external --data mature_outcomes.csv
```

For an isolated local smoke test of the checked-in demonstration bundle, the
operator must opt in explicitly:

```bash
python -m src.monitor_model --data data/credit_risk.csv --allow-unsigned-demo
python -m src.validate_external --data data/credit_risk.csv --allow-unsigned-demo
```

Never use `--allow-unsigned-demo` in a shared or production environment.

Drift output includes dataset and model digests, timestamps, row counts,
distribution drift, missing-rate changes, unknown-category rates, and separate
statuses. A newly generated model baseline records numeric missingness and its
reference population size. Legacy demo baselines show numeric missingness status
as unavailable because they predate those fields.

## Verification

Run the ML-focused suite independently of Django:

```bash
python -m unittest discover -s tests -v
```

It covers canonical feature construction, final-test isolation, missingness
drift, undefined fairness denominators, signed artifacts, report hashes,
version-tag matching, and the quick-release prohibition.
