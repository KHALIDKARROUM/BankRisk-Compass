from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_contract import (
    CATEGORICAL_FEATURES,
    EXCLUDED_LENDER_ASSIGNED_FEATURES,
    EXCLUDED_POLICY_FEATURES,
    FEATURE_CONTRACT_VERSION,
    FEATURES,
    FORM_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    model_feature_frame,
    with_canonical_derived_features,
)
from .model_reporting import (
    save_age_fairness_report,
    save_bootstrap_intervals,
    save_calibration_report,
)
from .release_artifacts import file_sha256, sign_manifest, snapshot_report_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "credit_risk.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_BUNDLE_PATH = MODELS_DIR / "credit_risk_model.pkl"
MODEL_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

RANDOM_STATE = 42
MODEL_VERSION = "2.2.0"
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1

RELEASE_REPORT_ARTIFACTS = {
    "business_report.md",
    "calibration_analysis.csv",
    "calibration_curve.png",
    "classification_report.txt",
    "confusion_matrix.png",
    "fairness_age_groups.csv",
    "final_model_metrics.csv",
    "metric_confidence_intervals.csv",
    "model_comparison.csv",
    "model_comparison.png",
    "permutation_importance.csv",
    "permutation_importance.png",
    "risk_band_validation.csv",
    "threshold_analysis.csv",
    "threshold_selection_validation.csv",
    "threshold_tradeoff.png",
    "threshold_uncertainty.csv",
}


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_credit_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.drop_duplicates().copy()

    # Extremely long employment values are data-quality issues in this dataset.
    data.loc[data["person_emp_length"] > 60, "person_emp_length"] = np.nan
    data.loc[data["person_age"] > 100, "person_age"] = np.nan

    # Production requests do not supply the source dataset's derived ratio.
    # Always reconstruct it from the same raw fields used by serving.
    return with_canonical_derived_features(data)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
    }


def evaluate_predictions(
    y_true: pd.Series,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "average_precision": average_precision_score(y_true, y_probability),
        "brier_score": brier_score_loss(y_true, y_probability),
    }


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | str]:
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, probabilities, threshold)
    return {"model": name, **metrics}


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, quick: bool) -> GridSearchCV:
    param_grid = {
        "classifier__n_estimators": [50] if quick else [200, 300],
        "classifier__max_depth": [16] if quick else [None, 16, 24],
        "classifier__min_samples_leaf": [2] if quick else [1, 2],
    }

    search = GridSearchCV(
        estimator=make_pipeline(
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
        param_grid=param_grid,
        cv=2 if quick else 3,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def build_threshold_table(
    y_true: pd.Series,
    y_probability: np.ndarray,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
    false_positive_cost: int = FALSE_POSITIVE_COST,
) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.01), 2):
        y_pred = (y_probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "business_cost": (false_negative_cost * fn) + (false_positive_cost * fp),
            }
        )

    return pd.DataFrame(rows)


def choose_business_threshold(threshold_table: pd.DataFrame) -> float:
    ranked = threshold_table.sort_values(
        ["business_cost", "f1_score", "recall"],
        ascending=[True, False, False],
    )
    return float(ranked.iloc[0]["threshold"])


def save_threshold_uncertainty(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_path: Path,
    *,
    iterations: int,
) -> pd.DataFrame:
    """Bootstrap the complete threshold-selection rule on validation data."""
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities)
    rng = np.random.default_rng(RANDOM_STATE)
    selections: list[dict[str, float]] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        sample_labels = labels[indices]
        if np.unique(sample_labels).size < 2:
            continue
        table = build_threshold_table(pd.Series(sample_labels), scores[indices])
        threshold = choose_business_threshold(table)
        row = table.loc[(table["threshold"] - threshold).abs().idxmin()]
        population = row["true_negatives"] + row["false_positives"] + row["false_negatives"] + row["true_positives"]
        selections.append(
            {
                "threshold": threshold,
                "business_cost": float(row["business_cost"]),
                "recall": float(row["recall"]),
                "review_rate": float((row["false_positives"] + row["true_positives"]) / population),
            }
        )
    samples = pd.DataFrame(selections)
    if samples.empty:
        raise RuntimeError("Threshold bootstrap produced no valid resamples.")
    summary: dict[str, Any] = {
        "evaluation_split": "threshold_selection",
        "bootstrap_iterations": len(samples),
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
    }
    for column in ["threshold", "business_cost", "recall", "review_rate"]:
        summary[f"{column}_median"] = float(samples[column].median())
        summary[f"{column}_ci_lower"] = float(samples[column].quantile(0.025))
        summary[f"{column}_ci_upper"] = float(samples[column].quantile(0.975))
    output = pd.DataFrame([summary])
    output.to_csv(output_path, index=False)
    return output


def build_threshold_reports(
    y_threshold: pd.Series,
    threshold_probability: np.ndarray,
    y_test: pd.Series,
    final_probability: np.ndarray,
    business_threshold: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build validation scenarios and one locked final-test operating point."""

    scenarios = build_threshold_table(y_threshold, threshold_probability).assign(
        evaluation_split="threshold_selection"
    )
    final_test_table = build_threshold_table(y_test, final_probability)
    final_row = final_test_table.iloc[
        (final_test_table["threshold"] - business_threshold).abs().argsort()[:1]
    ].iloc[0]
    return scenarios, final_row


def build_feature_reference(data: pd.DataFrame) -> dict[str, Any]:
    return {
        "numeric_medians": {
            column: float(data[column].median()) for column in FORM_NUMERIC_FEATURES
        },
        "categorical_modes": {
            column: str(data[column].mode(dropna=True).iloc[0])
            for column in CATEGORICAL_FEATURES
        },
        "categorical_options": {
            column: sorted(data[column].dropna().astype(str).unique().tolist())
            for column in CATEGORICAL_FEATURES
        },
        "numeric_bounds": {
            column: {
                "minimum": float(data[column].quantile(0.005)),
                "maximum": float(data[column].quantile(0.995)),
            }
            for column in FORM_NUMERIC_FEATURES
        },
    }


def build_drift_reference(data: pd.DataFrame) -> dict[str, Any]:
    numeric: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        values = data[column].dropna().astype(float)
        if values.empty:
            raise ValueError(f"Cannot build a drift baseline for all-missing feature: {column}")
        edges = np.unique(values.quantile(np.linspace(0, 1, 11)).to_numpy())
        if len(edges) < 3:
            edges = np.array([-np.inf, values.median(), np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
        counts, _ = np.histogram(values, bins=edges)
        proportions = (counts / counts.sum()).tolist()
        numeric[column] = {
            "edges": edges.tolist(),
            "proportions": proportions,
            "missing_count": int(data[column].isna().sum()),
            "missing_rate": float(data[column].isna().mean()),
        }

    categorical = {
        column: data[column].fillna("__MISSING__").astype(str).value_counts(normalize=True).to_dict()
        for column in CATEGORICAL_FEATURES
    }
    return {
        "numeric": numeric,
        "categorical": categorical,
        "observed_default_rate": float(data[TARGET].mean()),
        "reference_rows": int(len(data)),
    }


def build_score_reference(probabilities: np.ndarray) -> dict[str, Any]:
    """Build a training-only baseline for calibrated prediction drift."""
    values = pd.Series(np.asarray(probabilities, dtype=float)).dropna()
    if values.empty or not np.isfinite(values).all():
        raise ValueError("Cannot build a score baseline from missing or non-finite values.")
    edges = np.unique(values.quantile(np.linspace(0, 1, 11)).to_numpy())
    if len(edges) < 3:
        edges = np.array([-np.inf, values.median(), np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    counts, _ = np.histogram(values, bins=edges)
    return {
        "edges": edges.tolist(),
        "proportions": (counts / counts.sum()).tolist(),
        "mean": float(values.mean()),
        "reference_rows": int(len(values)),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def git_is_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def git_release_tag(expected_version: str = MODEL_VERSION) -> str:
    """Return the exact model-version tag pointing at HEAD."""
    try:
        tags = subprocess.check_output(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("A release must be built from a tagged Git commit.") from exc
    expected_tag = f"model-v{expected_version}"
    if tags != [expected_tag]:
        raise RuntimeError(
            f"Release tag must be exactly {expected_tag!r}; found {tags or 'no tags'}."
        )
    return expected_tag


def save_model_comparison_chart(results: pd.DataFrame) -> None:
    chart_data = results.set_index("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    ax = chart_data.plot(kind="bar", figsize=(11, 6), ylim=(0, 1), rot=0)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=160)
    plt.close()


def save_confusion_matrix_chart(
    y_test: pd.Series,
    y_probability: np.ndarray,
    threshold: float,
) -> None:
    y_pred = (y_probability >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-default", "Default"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix at Business Threshold ({threshold:.2f})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def save_threshold_chart(threshold_table: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision")
    ax1.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall")
    ax1.plot(threshold_table["threshold"], threshold_table["f1_score"], label="F1-score")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Metric")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(
        threshold_table["threshold"],
        threshold_table["business_cost"],
        color="#7a3e00",
        linestyle="--",
        label="Business Cost",
    )
    ax2.set_ylabel("Business Cost")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    plt.title("Threshold-Selection Tradeoff: Risk Capture vs Review Volume")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "threshold_tradeoff.png", dpi=160)
    plt.close(fig)


def save_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Measure the behavior of the calibrated predictor at its deployed policy cutoff."""
    sample_size = min(3000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
    y_sample = y_test.loc[X_sample.index]

    def deployed_policy_f1(estimator: Any, features: pd.DataFrame, labels: pd.Series) -> float:
        probabilities = estimator.predict_proba(features)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        return float(f1_score(labels, predictions, zero_division=0))

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring=deployed_policy_f1,
        n_repeats=7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    top = importance.head(10).sort_values("importance_mean")
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.title("Top Drivers of the Deployed Review Policy")
    plt.xlabel(f"Permutation importance (F1 decrease at {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "permutation_importance.png", dpi=160)
    plt.close()

    return importance


def save_risk_band_validation(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_path: Path,
    *,
    low_cutoff: float,
    medium_cutoff: float,
) -> pd.DataFrame:
    """Validate candidate display bands on threshold-selection data only."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    band_names = np.select(
        [scores <= low_cutoff, scores <= medium_cutoff],
        ["low", "medium"],
        default="high",
    )
    rows: list[dict[str, Any]] = []
    z = 1.96
    for name in ("low", "medium", "high"):
        mask = band_names == name
        count = int(mask.sum())
        if count == 0:
            raise RuntimeError(f"Candidate risk band {name!r} has no validation observations.")
        defaults = int(labels[mask].sum())
        rate = defaults / count
        denominator = 1 + (z**2 / count)
        centre = (rate + (z**2 / (2 * count))) / denominator
        half_width = (
            z
            * np.sqrt((rate * (1 - rate) / count) + (z**2 / (4 * count**2)))
            / denominator
        )
        rows.append(
            {
                "evaluation_split": "threshold_selection",
                "risk_band_policy_version": "validation-v1",
                "risk_band": name,
                "count": count,
                "population_share": count / len(labels),
                "defaults": defaults,
                "observed_default_rate": rate,
                "default_rate_ci_lower": max(0.0, centre - half_width),
                "default_rate_ci_upper": min(1.0, centre + half_width),
                "low_cutoff": low_cutoff,
                "medium_cutoff": medium_cutoff,
            }
        )
    result = pd.DataFrame(rows)
    rates = result["observed_default_rate"].to_numpy()
    if np.any(np.diff(rates) < 0):
        raise RuntimeError(
            "Candidate risk bands are not monotonic on threshold-selection data; "
            "release requires a revised, approved band policy."
        )
    result.to_csv(output_path, index=False)
    return result


def write_business_report(
    model_name: str,
    final_metrics: dict[str, float],
    business_metrics: dict[str, float],
    threshold: float,
    model_parameters: dict[str, Any],
    threshold_row: pd.Series,
    split_sizes: dict[str, int],
) -> None:
    report = f"""# Aegis-Credit Final Report

## Final Model

The final model is a calibrated, leakage-safe {model_name} classifier. Missing values, scaling, and one-hot encoding are fitted only on training data. Model selection, probability calibration, and threshold selection use separate data partitions; the final metrics below are measured once on an untouched test set.

Application-time scoring intentionally excludes lender-assigned fields (`loan_grade` and `loan_int_rate`) to avoid using information that may not exist when an applicant is first assessed.
Age is also excluded from the probability model; it is retained only for input plausibility checks and subgroup monitoring.

Data split:

- Training: {split_sizes["train"]:,} rows
- Model selection: {split_sizes["selection"]:,} rows
- Probability calibration: {split_sizes["calibration"]:,} rows
- Threshold selection: {split_sizes["threshold"]:,} rows
- Final test: {split_sizes["test"]:,} rows

Selected model parameters:

```text
{model_parameters}
```

## Default 0.50 Threshold Results

| Metric | Score |
|---|---:|
| Accuracy | {final_metrics["accuracy"]:.3f} |
| Precision | {final_metrics["precision"]:.3f} |
| Recall | {final_metrics["recall"]:.3f} |
| F1-score | {final_metrics["f1_score"]:.3f} |
| ROC-AUC | {final_metrics["roc_auc"]:.3f} |
| Average precision | {final_metrics["average_precision"]:.3f} |
| Brier score | {final_metrics["brier_score"]:.3f} |

## Business Threshold Results

The selected screening threshold is **{threshold:.2f}**. Its illustrative count-weighted objective assumes a false negative is 5x more costly than a false positive; it is not an approved expected-loss model:

- False negative: a risky borrower is approved.
- False positive: a safer borrower is unnecessarily routed to manual review.

| Metric | Score |
|---|---:|
| Accuracy | {business_metrics["accuracy"]:.3f} |
| Precision | {business_metrics["precision"]:.3f} |
| Recall | {business_metrics["recall"]:.3f} |
| F1-score | {business_metrics["f1_score"]:.3f} |
| ROC-AUC | {business_metrics["roc_auc"]:.3f} |
| Average precision | {business_metrics["average_precision"]:.3f} |
| Brier score | {business_metrics["brier_score"]:.3f} |
| False positives | {int(threshold_row["false_positives"])} |
| False negatives | {int(threshold_row["false_negatives"])} |
| Business cost | {int(threshold_row["business_cost"])} |

## Interpretation

The model is a decision-support tool, not an autonomous approval system. The business threshold prioritizes recall for defaults while monitoring the number of safer applicants routed to review. Age-group diagnostics and calibration reports are generated for governance review, but they do not replace a complete fair-lending assessment using legally appropriate protected-class data.
"""
    (REPORTS_DIR / "business_report.md").write_text(report, encoding="utf-8")


def train_and_save(
    quick: bool = False,
    require_clean: bool = False,
    release: bool = False,
    demo: bool = False,
) -> dict[str, Any]:
    """Train either a signed release or an explicitly local-only demo artifact.

    A demo build is intentionally unsigned and is never considered an approved
    release, regardless of the data being used.  It exists so the portfolio can
    ship a current, reproducible UI artifact without publishing unsupported
    lending-performance claims.
    """

    if release == demo:
        raise RuntimeError("Choose exactly one artifact mode: `--release` or `--demo`.")
    if quick and release:
        raise RuntimeError(
            "Quick training is for local experimentation and cannot produce a release artifact."
        )
    if release and git_is_dirty() is not False:
        raise RuntimeError(
            "Refusing to produce a release artifact from a dirty Git worktree."
        )
    if release and os.getenv("DATA_PROVENANCE_VERIFIED", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("Refusing to train a release from data without verified provenance approval.")
    release_tag = git_release_tag(MODEL_VERSION) if release else "local-demo"
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    data = load_credit_data()
    X = model_feature_frame(data)
    y = data[TARGET]

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )
    X_selection, X_calibration_threshold, y_selection, y_calibration_threshold = (
        train_test_split(
            X_holdout,
            y_holdout,
            test_size=0.60,
            random_state=RANDOM_STATE,
            stratify=y_holdout,
        )
    )
    X_calibration, X_threshold, y_calibration, y_threshold = train_test_split(
        X_calibration_threshold,
        y_calibration_threshold,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_calibration_threshold,
    )

    grid_search = tune_random_forest(X_train, y_train, quick=quick)
    candidate_models = build_candidate_models()
    candidate_models["Random Forest"] = grid_search.best_estimator_
    candidate_artifacts: dict[str, tuple[Pipeline, Any, float]] = {}
    model_results = []
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        calibrated_candidate = CalibratedClassifierCV(
            FrozenEstimator(model),
            method="sigmoid",
        )
        calibrated_candidate.fit(X_calibration, y_calibration)
        threshold_probability = calibrated_candidate.predict_proba(X_threshold)[:, 1]
        candidate_threshold = choose_business_threshold(
            build_threshold_table(y_threshold, threshold_probability)
        )
        selection_probability = calibrated_candidate.predict_proba(X_selection)[:, 1]
        selection_threshold_table = build_threshold_table(
            y_selection,
            selection_probability,
        )
        selection_row = selection_threshold_table.iloc[
            (selection_threshold_table["threshold"] - candidate_threshold).abs().argsort()[:1]
        ].iloc[0]
        result = {
            "model": name,
            **evaluate_predictions(
                y_selection,
                selection_probability,
                threshold=candidate_threshold,
            ),
            "decision_threshold": candidate_threshold,
            "business_cost": int(selection_row["business_cost"]),
            "evaluation_split": "selection_after_calibration_and_thresholding",
        }
        model_results.append(result)
        candidate_artifacts[name] = (model, calibrated_candidate, candidate_threshold)

    results = pd.DataFrame(model_results).sort_values(
        ["business_cost", "f1_score", "roc_auc"],
        ascending=[True, False, False],
    )
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    save_model_comparison_chart(results)

    model_name = str(results.iloc[0]["model"])
    explanation_pipeline, calibrated_model, business_threshold = candidate_artifacts[model_name]
    classifier = explanation_pipeline.named_steps["classifier"]
    selected_parameter_names = (
        ("n_estimators", "learning_rate", "max_depth", "min_samples_leaf")
        if model_name == "Gradient Boosting"
        else ("n_estimators", "max_depth", "min_samples_leaf", "class_weight")
        if model_name == "Random Forest"
        else ("C", "class_weight", "max_iter")
    )
    classifier_parameters = classifier.get_params()
    selected_model_parameters = {
        name: classifier_parameters[name]
        for name in selected_parameter_names
        if name in classifier_parameters
    }
    threshold_probability = calibrated_model.predict_proba(X_threshold)[:, 1]
    final_probability = calibrated_model.predict_proba(X_test)[:, 1]
    score_reference = build_score_reference(
        calibrated_model.predict_proba(X_train)[:, 1]
    )
    final_metrics = evaluate_predictions(y_test, final_probability, threshold=0.5)
    # The final test is measured only at preselected thresholds. Interactive
    # threshold scenarios must remain on the dedicated threshold partition.
    threshold_selection_table, threshold_row = build_threshold_reports(
        y_threshold,
        threshold_probability,
        y_test,
        final_probability,
        business_threshold,
    )
    business_metrics = evaluate_predictions(
        y_test,
        final_probability,
        threshold=business_threshold,
    )
    threshold_selection_table.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    threshold_selection_table.to_csv(
        REPORTS_DIR / "threshold_selection_validation.csv",
        index=False,
    )
    save_threshold_uncertainty(
        y_threshold,
        threshold_probability,
        REPORTS_DIR / "threshold_uncertainty.csv",
        iterations=200 if quick else 1000,
    )
    pd.DataFrame(
        [
            {
                "model": model_name,
                "evaluation_split": "final_test",
                "decision_threshold": 0.5,
                **final_metrics,
            },
            {
                "model": model_name,
                "evaluation_split": "final_test",
                "decision_threshold": business_threshold,
                **business_metrics,
            },
        ]
    ).to_csv(REPORTS_DIR / "final_model_metrics.csv", index=False)

    final_pred = (final_probability >= business_threshold).astype(int)
    (REPORTS_DIR / "classification_report.txt").write_text(
        classification_report(
            y_test,
            final_pred,
            target_names=["Non-default", "Default"],
            digits=3,
        ),
        encoding="utf-8",
    )

    save_confusion_matrix_chart(y_test, final_probability, business_threshold)
    save_threshold_chart(threshold_selection_table)
    permutation = save_permutation_importance(
        calibrated_model,
        X_test,
        y_test,
        business_threshold,
    )
    calibration = save_calibration_report(
        y_test,
        final_probability,
        REPORTS_DIR,
    )
    fairness = save_age_fairness_report(
        data,
        y_test,
        final_probability,
        business_threshold,
        REPORTS_DIR,
    )
    confidence_intervals = save_bootstrap_intervals(
        y_test,
        final_probability,
        business_threshold,
        REPORTS_DIR,
        iterations=200 if quick else 1000,
    )
    risk_band_policy = {
        "policy_version": "validation-v1",
        "low": round(max(0.05, business_threshold * 0.5), 2),
        "medium": round(
            min(0.90, max(business_threshold + 0.10, business_threshold * 2)),
            2,
        ),
    }
    save_risk_band_validation(
        y_threshold,
        threshold_probability,
        REPORTS_DIR / "risk_band_validation.csv",
        low_cutoff=risk_band_policy["low"],
        medium_cutoff=risk_band_policy["medium"],
    )
    split_sizes = {
        "train": len(X_train),
        "selection": len(X_selection),
        "calibration": len(X_calibration),
        "threshold": len(X_threshold),
        "test": len(X_test),
    }
    write_business_report(
        model_name,
        final_metrics,
        business_metrics,
        business_threshold,
        selected_model_parameters,
        threshold_row,
        split_sizes,
    )

    bundle = {
        "pipeline": explanation_pipeline,
        "predictor": calibrated_model,
        "model_name": model_name,
        "threshold": business_threshold,
        "default_threshold_metrics": final_metrics,
        "business_threshold_metrics": business_metrics,
        "best_params": selected_model_parameters,
        "random_forest_best_params": grid_search.best_params_,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "feature_reference": build_feature_reference(data.loc[X_train.index]),
        "drift_reference": build_drift_reference(data.loc[X_train.index]),
        "score_reference": score_reference,
        "permutation_importance": permutation,
        "calibration_analysis": calibration,
        "fairness_age_groups": fairness,
        "metric_confidence_intervals": confidence_intervals,
        "excluded_lender_assigned_features": EXCLUDED_LENDER_ASSIGNED_FEATURES,
        "excluded_policy_features": EXCLUDED_POLICY_FEATURES,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "model_version": MODEL_VERSION,
        "training_mode": "full_release" if release else "synthetic_demo",
        "random_state": RANDOM_STATE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": file_sha256(DATA_PATH),
        "git_commit": git_commit(),
        "git_dirty": False if release else git_is_dirty(),
        "git_tag": release_tag,
        "data_provenance_verified": bool(release),
        "split_sizes": split_sizes,
        "cost_assumptions": {
            "false_negative": FALSE_NEGATIVE_COST,
            "false_positive": FALSE_POSITIVE_COST,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "risk_bands": risk_band_policy,
    }

    if demo:
        # The demo is intentionally a conventional local artifact. It is
        # hash-checked by the runtime but cannot be promoted as a release.
        joblib.dump(bundle, MODEL_BUNDLE_PATH, compress=3)
        manifest = {
            "model_version": MODEL_VERSION,
            "model_name": model_name,
            "trained_at_utc": bundle["trained_at_utc"],
            "model_sha256": file_sha256(MODEL_BUNDLE_PATH),
            "data_sha256": bundle["data_sha256"],
            "git_commit": bundle["git_commit"],
            "git_dirty": bundle["git_dirty"],
            "git_tag": "local-demo",
            "data_provenance_verified": False,
            "dataset_kind": "synthetic_demo",
            "split_sizes": bundle["split_sizes"],
            "cost_assumptions": bundle["cost_assumptions"],
            "risk_bands": bundle["risk_bands"],
            "best_params": bundle["best_params"],
            "runtime_versions": bundle["runtime_versions"],
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "training_mode": bundle["training_mode"],
            "random_state": bundle["random_state"],
            "features": FEATURES,
            "excluded_lender_assigned_features": EXCLUDED_LENDER_ASSIGNED_FEATURES,
            "excluded_policy_features": EXCLUDED_POLICY_FEATURES,
            "threshold": business_threshold,
        }
        MODEL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {
            "model_path": str(MODEL_BUNDLE_PATH),
            "model_comparison": results,
            "default_metrics": final_metrics,
            "business_metrics": business_metrics,
            "business_threshold": business_threshold,
            "model_name": model_name,
        }

    # Release bundles are content-addressed and never overwritten. The manifest
    # is signed only after the immutable artifact path and digest are known.
    staging_path = MODELS_DIR / ".model-release-staging.pkl"
    joblib.dump(bundle, staging_path, compress=3)
    model_sha256 = file_sha256(staging_path)
    release_path = MODELS_DIR / "releases" / model_sha256 / "model.pkl"
    release_path.parent.mkdir(parents=True, exist_ok=True)
    if release_path.exists():
        if file_sha256(release_path) != model_sha256:
            raise RuntimeError("Refusing to overwrite an existing immutable model release path.")
        staging_path.unlink()
    else:
        staging_path.replace(release_path)
    report_bundle = snapshot_report_bundle(
        reports_dir=REPORTS_DIR,
        release_dir=release_path.parent,
        artifact_names=RELEASE_REPORT_ARTIFACTS,
        models_dir=MODELS_DIR,
    )
    manifest = sign_manifest(
        {
                "model_version": MODEL_VERSION,
                "model_name": model_name,
                "trained_at_utc": bundle["trained_at_utc"],
                "model_sha256": model_sha256,
                "artifact_path": release_path.relative_to(MODELS_DIR).as_posix(),
                "data_sha256": bundle["data_sha256"],
                "git_commit": bundle["git_commit"],
                "git_dirty": bundle["git_dirty"],
                "git_tag": bundle["git_tag"],
                "data_provenance_verified": True,
                "split_sizes": bundle["split_sizes"],
                "cost_assumptions": bundle["cost_assumptions"],
                "risk_bands": bundle["risk_bands"],
                "best_params": bundle["best_params"],
                "runtime_versions": bundle["runtime_versions"],
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                "training_mode": bundle["training_mode"],
                "random_state": bundle["random_state"],
                "features": FEATURES,
                "excluded_lender_assigned_features": EXCLUDED_LENDER_ASSIGNED_FEATURES,
                "excluded_policy_features": EXCLUDED_POLICY_FEATURES,
                "threshold": business_threshold,
                "report_bundle": report_bundle,
        }
    )
    manifest_staging = MODEL_MANIFEST_PATH.with_suffix(".json.tmp")
    manifest_staging.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_staging.replace(MODEL_MANIFEST_PATH)

    return {
        "model_path": str(release_path),
        "model_comparison": results,
        "default_metrics": final_metrics,
        "business_metrics": business_metrics,
        "business_threshold": business_threshold,
        "model_name": model_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Aegis-Credit model.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Request quick mode (explicitly rejected for signed release builds).",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Build a signed immutable release from a clean, tagged commit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build an unsigned local-only synthetic demonstration artifact.",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail when Git has uncommitted changes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = train_and_save(
        quick=args.quick,
        require_clean=args.require_clean,
        release=args.release,
        demo=args.demo,
    )
    print(f"Saved model bundle: {output['model_path']}")
    print(f"Business threshold: {output['business_threshold']:.2f}")
    print(f"Selected model: {output['model_name']}")
    print("Default-threshold metrics:")
    for metric, value in output["default_metrics"].items():
        print(f"  {metric}: {value:.3f}")
