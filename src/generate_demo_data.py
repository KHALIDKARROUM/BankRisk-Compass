"""Generate the repository's deterministic, synthetic demonstration data.

The dashboard deliberately ships no customer or third-party credit data.  This
generator is the authoritative source for ``data/credit_risk.csv`` and makes
the dataset reproducible for reviewers without downloading or redistributing
an unlicensed file.  The outcomes are simulated and must never be interpreted
as real repayment performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "credit_risk.csv"
DEFAULT_ROWS = 6_000
SEED = 20_260_830


def build_demo_dataset(rows: int = DEFAULT_ROWS, *, seed: int = SEED) -> pd.DataFrame:
    """Return a deterministic, statistically plausible synthetic data frame."""

    if rows < 1_000:
        raise ValueError("At least 1,000 rows are required for the demonstration dataset.")

    random = np.random.default_rng(seed)
    age = random.integers(21, 71, size=rows)
    employment = np.minimum(
        random.gamma(shape=2.2, scale=3.1, size=rows),
        np.maximum(age - 16, 0),
    ).round(1)
    income = np.clip(
        random.lognormal(mean=11.0, sigma=0.48, size=rows) * (0.78 + age / 100),
        18_000,
        350_000,
    ).round(-2).astype(int)
    home = random.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
        size=rows,
        p=[0.48, 0.34, 0.15, 0.03],
    )
    intent = random.choice(
        [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE",
        ],
        size=rows,
        p=[0.29, 0.16, 0.11, 0.15, 0.17, 0.12],
    )
    credit_history = np.minimum(
        random.gamma(shape=2.3, scale=3.0, size=rows),
        np.maximum(age - 16, 1),
    ).round().astype(int)
    previous_default = random.choice(["N", "Y"], size=rows, p=[0.83, 0.17])
    loan_amount = np.clip(
        income * random.uniform(0.035, 0.65, size=rows),
        500,
        50_000,
    ).round(-2).astype(int)
    loan_to_income = loan_amount / income

    rent_penalty = np.where(home == "RENT", 0.32, np.where(home == "OTHER", 0.18, 0.0))
    intent_penalty = np.select(
        [intent == "MEDICAL", intent == "VENTURE", intent == "DEBTCONSOLIDATION"],
        [0.25, 0.17, 0.12],
        default=0.0,
    )
    linear_risk = (
        -3.25
        + 4.25 * loan_to_income
        + 1.05 * (previous_default == "Y")
        + rent_penalty
        + intent_penalty
        - 0.045 * employment
        - 0.035 * credit_history
        + random.normal(0, 0.42, size=rows)
    )
    default_probability = 1 / (1 + np.exp(-linear_risk))
    status = random.binomial(1, default_probability)

    interest_rate = np.clip(
        5.6 + 10.8 * default_probability + random.normal(0, 1.25, size=rows),
        5.4,
        22.0,
    ).round(2)
    grade = np.select(
        [
            default_probability < 0.08,
            default_probability < 0.14,
            default_probability < 0.22,
            default_probability < 0.32,
            default_probability < 0.46,
        ],
        ["A", "B", "C", "D", "E"],
        default="F",
    )

    return pd.DataFrame(
        {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home,
            "person_emp_length": employment,
            "loan_intent": intent,
            "loan_grade": grade,
            "loan_amnt": loan_amount,
            "loan_int_rate": interest_rate,
            "loan_status": status,
            # Deliberately serialize a rounded source field. Training and serving
            # overwrite it from raw amount and income through feature_contract.
            "loan_percent_income": np.round(loan_to_income, 2),
            "cb_person_default_on_file": previous_default,
            "cb_person_cred_hist_length": credit_history,
        }
    )


def write_demo_dataset(output: Path = DEFAULT_OUTPUT, *, rows: int = DEFAULT_ROWS) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    build_demo_dataset(rows).to_csv(output, index=False, lineterminator="\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Aegis-Credit demo data.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    path = write_demo_dataset(arguments.output, rows=arguments.rows)
    print(f"Wrote deterministic synthetic demonstration data: {path}")
