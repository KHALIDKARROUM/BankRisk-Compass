from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURES,
    FeatureContractError,
    model_feature_frame,
    validate_binary_target,
)
from src.model_reporting import save_age_fairness_report
from src.monitor_model import build_drift_report
from src.release_artifacts import (
    ArtifactIntegrityError,
    file_sha256,
    load_verified_model_artifact,
    sign_manifest,
    snapshot_report_bundle,
    verified_report_paths,
)
from src.train_model import (
    build_drift_reference,
    build_score_reference,
    build_threshold_reports,
    git_release_tag,
    load_credit_data,
    save_risk_band_validation,
    train_and_save,
)
from src.validate_external import evaluate_external_data


class CapturingPredictor:
    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        self.seen = data.copy()
        probability = np.array([0.2, 0.8])[: len(data)]
        return np.column_stack([1 - probability, probability])


def raw_applications() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "person_age": 30,
                "person_income": 50_000,
                "person_emp_length": 5.0,
                "loan_amnt": 10_000,
                "loan_percent_income": 0.99,
                "cb_person_cred_hist_length": 6,
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "cb_person_default_on_file": "N",
                "loan_status": 0,
            },
            {
                "person_age": 42,
                "person_income": 80_000,
                "person_emp_length": 12.0,
                "loan_amnt": 20_000,
                "loan_percent_income": 0.01,
                "cb_person_cred_hist_length": 15,
                "person_home_ownership": "MORTGAGE",
                "loan_intent": "MEDICAL",
                "cb_person_default_on_file": "Y",
                "loan_status": 1,
            },
        ]
    )


class FeatureContractTests(unittest.TestCase):
    def test_model_frame_overwrites_untrusted_derived_ratio(self) -> None:
        frame = model_feature_frame(raw_applications())
        self.assertEqual(frame.columns.tolist(), FEATURES)
        self.assertEqual(frame["loan_percent_income"].tolist(), [0.2, 0.25])

    def test_nonpositive_income_is_rejected(self) -> None:
        data = raw_applications()
        data.loc[0, "person_income"] = 0
        with self.assertRaisesRegex(FeatureContractError, "greater than zero"):
            model_feature_frame(data)

    def test_non_numeric_raw_feature_is_rejected(self) -> None:
        data = raw_applications()
        data["person_emp_length"] = data["person_emp_length"].astype(object)
        data.loc[0, "person_emp_length"] = "unknown"
        with self.assertRaisesRegex(FeatureContractError, "person_emp_length"):
            model_feature_frame(data)

    def test_binary_target_requires_both_mature_classes(self) -> None:
        with self.assertRaisesRegex(FeatureContractError, "both binary"):
            validate_binary_target(pd.Series([0, 0, 0]))

    def test_training_loader_uses_canonical_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.csv"
            raw_applications().to_csv(path, index=False)
            loaded = load_credit_data(path)
        self.assertEqual(loaded["loan_percent_income"].tolist(), [0.2, 0.25])


class EvaluationIsolationTests(unittest.TestCase):
    def test_threshold_scenarios_are_built_from_validation_only(self) -> None:
        y_threshold = pd.Series([0, 0, 1, 1])
        threshold_probability = np.array([0.1, 0.2, 0.8, 0.9])
        y_test = pd.Series([0, 1, 1, 1, 1, 1])
        final_probability = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

        scenarios, final_row = build_threshold_reports(
            y_threshold,
            threshold_probability,
            y_test,
            final_probability,
            0.5,
        )

        scenario_population = (
            scenarios[["true_negatives", "false_positives", "false_negatives", "true_positives"]]
            .sum(axis=1)
            .unique()
            .tolist()
        )
        self.assertEqual(scenario_population, [len(y_threshold)])
        self.assertTrue(scenarios["evaluation_split"].eq("threshold_selection").all())
        self.assertEqual(
            int(
                final_row["true_negatives"]
                + final_row["false_positives"]
                + final_row["false_negatives"]
                + final_row["true_positives"]
            ),
            len(y_test),
        )

    def test_external_validation_uses_canonical_serving_features(self) -> None:
        predictor = CapturingPredictor()
        data = raw_applications()
        bundle = {
            "predictor": predictor,
            "pipeline": predictor,
            "threshold": 0.5,
            "feature_reference": {
                "categorical_options": {
                    feature: sorted(data[feature].unique().tolist())
                    for feature in (
                        "person_home_ownership",
                        "loan_intent",
                        "cb_person_default_on_file",
                    )
                }
            },
        }
        metrics = evaluate_external_data(data, bundle)
        self.assertIsNotNone(predictor.seen)
        assert predictor.seen is not None
        self.assertEqual(predictor.seen["loan_percent_income"].tolist(), [0.2, 0.25])
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_risk_band_evidence_is_validation_only_and_monotonic(self) -> None:
        truth = pd.Series([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.05, 0.10, 0.20, 0.40, 0.70, 0.90])
        with tempfile.TemporaryDirectory() as temporary:
            report = save_risk_band_validation(
                truth,
                probabilities,
                Path(temporary) / "risk-bands.csv",
                low_cutoff=0.15,
                medium_cutoff=0.60,
            )

        self.assertTrue(report["evaluation_split"].eq("threshold_selection").all())
        self.assertEqual(report["risk_band"].tolist(), ["low", "medium", "high"])
        self.assertTrue(report["observed_default_rate"].is_monotonic_increasing)


class MonitoringTests(unittest.TestCase):
    def test_numeric_missingness_drift_is_reported(self) -> None:
        baseline = raw_applications()
        baseline_features = model_feature_frame(baseline)
        baseline_features["loan_status"] = baseline["loan_status"]
        bundle = {
            "model_version": "test",
            "drift_reference": build_drift_reference(baseline_features),
        }
        incoming = pd.concat([baseline_features.drop(columns="loan_status")] * 2, ignore_index=True)
        incoming.loc[:1, "person_emp_length"] = np.nan

        report = build_drift_report(
            incoming,
            bundle,
            dataset="unit-test",
            generated_at_utc="2026-01-01T00:00:00+00:00",
        )
        row = report.loc[report["feature"].eq("person_emp_length")].iloc[0]
        self.assertEqual(row["actual_missing_rate"], 0.5)
        self.assertEqual(row["missing_rate_delta"], 0.5)
        self.assertEqual(row["missingness_status"], "drift")
        self.assertEqual(row["status"], "drift")
        self.assertEqual(row["dataset"], "unit-test")

    def test_calibrated_score_drift_is_reported_when_baseline_is_available(self) -> None:
        baseline_probabilities = np.array([0.1, 0.2, 0.7, 0.8])

        class ScorePredictor:
            def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
                probabilities = np.full(len(data), 0.95)
                return np.column_stack([1 - probabilities, probabilities])

        baseline = raw_applications()
        features = model_feature_frame(baseline)
        features["loan_status"] = baseline["loan_status"]
        bundle = {
            "model_version": "test",
            "drift_reference": build_drift_reference(features),
            "score_reference": build_score_reference(baseline_probabilities),
            "predictor": ScorePredictor(),
        }
        report = build_drift_report(model_feature_frame(baseline), bundle)
        score_row = report.loc[report["feature"].eq("model_score")].iloc[0]

        self.assertEqual(score_row["feature_type"], "prediction")
        self.assertAlmostEqual(score_row["actual_score_mean"], 0.95)
        self.assertIn(score_row["status"], {"watch", "drift"})


class FairnessReportingTests(unittest.TestCase):
    def test_undefined_subgroup_rate_remains_nan_and_has_intervals(self) -> None:
        data = pd.DataFrame({"person_age": [20, 21, 22]})
        truth = pd.Series([0, 0, 0])
        probabilities = np.array([0.1, 0.8, 0.2])
        with tempfile.TemporaryDirectory() as temporary:
            report = save_age_fairness_report(
                data,
                truth,
                probabilities,
                0.5,
                Path(temporary),
            )
        row = report.iloc[0]
        self.assertTrue(np.isnan(row["true_positive_rate"]))
        self.assertTrue(np.isnan(row["true_positive_rate_lower_95"]))
        self.assertIn("false_positive_rate_lower_95", report.columns)


class ReleaseArtifactTests(unittest.TestCase):
    @staticmethod
    def _raw_private_key(private_key: Ed25519PrivateKey) -> str:
        raw = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return base64.b64encode(raw).decode("ascii")

    @staticmethod
    def _raw_public_key(private_key: Ed25519PrivateKey) -> str:
        raw = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw).decode("ascii")

    def test_unsigned_demo_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            model_path = models_dir / "credit_risk_model.pkl"
            joblib.dump({"model_version": "demo"}, model_path)
            manifest_path = models_dir / "model_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "model_version": "demo",
                        "model_sha256": file_sha256(model_path),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ArtifactIntegrityError, "Unsigned demonstration"):
                load_verified_model_artifact(manifest_path, demo_model_path=model_path)
            loaded = load_verified_model_artifact(
                manifest_path,
                demo_model_path=model_path,
                allow_unsigned_demo=True,
            )
            self.assertFalse(loaded.signed_release)

    def test_signed_model_and_report_bundle_are_verified(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models_dir = root / "models"
            reports_dir = root / "reports"
            models_dir.mkdir()
            reports_dir.mkdir()
            staging = root / "staging.pkl"
            bundle = {
                "model_version": "9.9.9",
                "git_dirty": False,
                "git_tag": "model-v9.9.9",
                "data_sha256": "data-digest",
                "feature_contract_version": "test-contract",
                "features": ["feature"],
                "threshold": 0.25,
            }
            joblib.dump(bundle, staging)
            model_hash = file_sha256(staging)
            release_dir = models_dir / "releases" / model_hash
            release_dir.mkdir(parents=True)
            model_path = release_dir / "model.pkl"
            staging.replace(model_path)
            (reports_dir / "metrics.csv").write_text("metric,value\nauc,0.8\n", encoding="utf-8")
            report_bundle = snapshot_report_bundle(
                reports_dir=reports_dir,
                release_dir=release_dir,
                artifact_names={"metrics.csv"},
                models_dir=models_dir,
            )
            manifest = sign_manifest(
                {
                    "model_version": "9.9.9",
                    "model_sha256": model_hash,
                    "artifact_path": model_path.relative_to(models_dir).as_posix(),
                    "data_sha256": "data-digest",
                    "git_dirty": False,
                    "git_tag": "model-v9.9.9",
                    "data_provenance_verified": True,
                    "feature_contract_version": "test-contract",
                    "features": ["feature"],
                    "threshold": 0.25,
                    "report_bundle": report_bundle,
                },
                private_key_base64=self._raw_private_key(private_key),
                key_id="unit-test",
            )
            manifest_path = models_dir / "model_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            verified = load_verified_model_artifact(
                manifest_path,
                public_key_base64=self._raw_public_key(private_key),
            )
            self.assertTrue(verified.signed_release)
            report_paths = verified_report_paths(manifest, models_dir=models_dir)
            self.assertEqual(report_paths["metrics.csv"].read_text(encoding="utf-8"), "metric,value\nauc,0.8\n")
            report_paths["metrics.csv"].write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactIntegrityError, "metrics.csv"):
                verified_report_paths(manifest, models_dir=models_dir)

    @patch("src.train_model.subprocess.check_output", return_value="model-v2.2.0\n")
    def test_release_tag_must_match_model_version(self, _check_output) -> None:
        self.assertEqual(git_release_tag("2.2.0"), "model-v2.2.0")

    @patch("src.train_model.subprocess.check_output", return_value="model-v1.0.0\n")
    def test_mismatched_release_tag_is_rejected(self, _check_output) -> None:
        with self.assertRaisesRegex(RuntimeError, "model-v2.2.0"):
            git_release_tag("2.2.0")

    def test_quick_release_is_rejected_before_training(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot produce a release"):
            train_and_save(quick=True, release=True)

    def test_demo_and_release_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly one artifact mode"):
            train_and_save(release=False, demo=False)


class CurrentArtifactContractTests(unittest.TestCase):
    def test_checked_in_demo_manifest_declares_the_current_contract(self) -> None:
        manifest = json.loads(
            (Path(__file__).resolve().parents[1] / "models" / "model_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["feature_contract_version"], FEATURE_CONTRACT_VERSION)
        self.assertEqual(manifest["dataset_kind"], "synthetic_demo")


if __name__ == "__main__":
    unittest.main()
