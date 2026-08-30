# Demonstration Dataset Provenance

`data/credit_risk.csv` is a **deterministic synthetic demonstration dataset**.
It is generated entirely within this repository by
[`src/generate_demo_data.py`](src/generate_demo_data.py), using a fixed seed
and no external downloads, customer data, or third-party source files.

## Redistribution license

The generated CSV and its generator are dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The repository's
application source remains under the MIT License. See
[`data/LICENSE.md`](data/LICENSE.md) for the dedicated data notice.

## Reproducibility

Regenerate the checked-in file from the project root:

```bash
python -m src.generate_demo_data
```

The file is deterministic for the checked-in generator version, row count, and
seed. The model manifest binds the exact generated-file SHA-256 to the
unsigned local-only demo artifact.

## Operational prohibition

The variables and outcomes are simulated. They do not represent an actual
lender, product, geography, population, approval policy, repayment event, or
performance window. They must not be used for lending decisions, performance
claims, fairness conclusions, or a production model release.

Any prospective production dataset requires a separate approved provenance
record covering its authoritative source, license and permission, collection
period, population, field definitions, de-identification/privacy assessment,
outcome maturity, and exact digest. `DATA_PROVENANCE_VERIFIED=True` is reserved
for that independently approved evidence; it does not approve this demo data.
