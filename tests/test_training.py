from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nostrain.model import ModelState
from nostrain.runtime import NUMPY_TRAINING_BACKEND, PYTHON_TRAINING_BACKEND
from nostrain.training import (
    MLP_REGRESSION_RUNTIME,
    LateGradientRecord,
    LateGradientReconciliationSummary,
    LinearRegressionDataset,
    LocalTrainingConfig,
    RegressionDataset,
    TrainingCheckpoint,
    TrainingRoundSummary,
    evaluate_regression,
    evaluate_linear_regression,
    initialize_mlp_regression_state,
    train_regression,
    train_linear_regression,
)
from tests.helpers import assert_model_state_almost_equal


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TrainingRuntimeTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        src_path = str(ROOT / "src")
        env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
        return subprocess.run(
            [sys.executable, "-m", "nostrain", *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_train_linear_regression_reduces_loss(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")
        dataset = LinearRegressionDataset.from_path(FIXTURES / "linear_dataset_worker_a.json")

        result = train_linear_regression(
            state,
            dataset,
            config=LocalTrainingConfig(
                steps=40,
                learning_rate=0.05,
                batch_size=2,
            ),
        )

        self.assertLess(result.loss_after, result.loss_before)
        self.assertEqual(result.trained_state.parameter_count, 2)
        self.assertEqual(dataset.feature_count, 2)
        self.assertAlmostEqual(
            evaluate_linear_regression(result.trained_state, dataset),
            result.loss_after,
            places=8,
        )

    def test_train_mlp_regression_reduces_loss(self) -> None:
        state = ModelState.from_path(FIXTURES / "mlp_initial_state.json")
        dataset = RegressionDataset.from_path(FIXTURES / "mlp_dataset_worker_a.json")

        result = train_regression(
            state,
            dataset,
            config=LocalTrainingConfig(
                steps=40,
                learning_rate=0.05,
                batch_size=2,
            ),
            runtime_name=MLP_REGRESSION_RUNTIME,
        )

        self.assertLess(result.loss_after, result.loss_before)
        self.assertEqual(result.runtime_name, MLP_REGRESSION_RUNTIME)
        self.assertEqual(result.trained_state.parameter_count, 4)
        self.assertAlmostEqual(
            evaluate_regression(
                result.trained_state,
                dataset,
                runtime_name=MLP_REGRESSION_RUNTIME,
            ),
            result.loss_after,
            places=8,
        )

    def test_train_linear_regression_numpy_backend_matches_python_backend(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")
        dataset = LinearRegressionDataset.from_path(FIXTURES / "linear_dataset_worker_a.json")
        config = LocalTrainingConfig(
            steps=40,
            learning_rate=0.05,
            batch_size=2,
        )

        python_result = train_linear_regression(
            state,
            dataset,
            config=config,
            backend_name=PYTHON_TRAINING_BACKEND,
        )
        numpy_result = train_linear_regression(
            state,
            dataset,
            config=config,
            backend_name=NUMPY_TRAINING_BACKEND,
        )

        self.assertEqual(numpy_result.backend_name, NUMPY_TRAINING_BACKEND)
        self.assertAlmostEqual(python_result.loss_after, numpy_result.loss_after, places=10)
        assert_model_state_almost_equal(
            self,
            python_result.trained_state,
            numpy_result.trained_state,
            places=10,
        )

    def test_train_mlp_regression_numpy_backend_matches_python_backend(self) -> None:
        state = ModelState.from_path(FIXTURES / "mlp_initial_state.json")
        dataset = RegressionDataset.from_path(FIXTURES / "mlp_dataset_worker_a.json")
        config = LocalTrainingConfig(
            steps=40,
            learning_rate=0.05,
            batch_size=2,
        )

        python_result = train_regression(
            state,
            dataset,
            config=config,
            runtime_name=MLP_REGRESSION_RUNTIME,
            backend_name=PYTHON_TRAINING_BACKEND,
        )
        numpy_result = train_regression(
            state,
            dataset,
            config=config,
            runtime_name=MLP_REGRESSION_RUNTIME,
            backend_name=NUMPY_TRAINING_BACKEND,
        )

        self.assertEqual(numpy_result.backend_name, NUMPY_TRAINING_BACKEND)
        self.assertAlmostEqual(python_result.loss_after, numpy_result.loss_after, places=8)
        assert_model_state_almost_equal(
            self,
            python_result.trained_state,
            numpy_result.trained_state,
            places=8,
        )

    def test_initialize_mlp_state_is_deterministic(self) -> None:
        first = initialize_mlp_regression_state(2, 4, seed=7, weight_scale=0.3)
        second = initialize_mlp_regression_state(2, 4, seed=7, weight_scale=0.3)

        self.assertEqual(first.to_json_obj(), second.to_json_obj())

    def test_cli_train_local_writes_state_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tempdir = Path(temporary_directory)
            trained_state = tempdir / "trained-state.json"
            metrics = tempdir / "metrics.json"

            self._run(
                "train-local",
                str(FIXTURES / "linear_initial_state.json"),
                str(FIXTURES / "linear_dataset_worker_a.json"),
                "--steps",
                "30",
                "--learning-rate",
                "0.05",
                "--batch-size",
                "2",
                "--metrics-out",
                str(metrics),
                "-o",
                str(trained_state),
            )

            trained_json = json.loads(trained_state.read_text(encoding="utf-8"))
            metrics_json = json.loads(metrics.read_text(encoding="utf-8"))

            self.assertIn("parameters", trained_json)
            self.assertLess(metrics_json["loss_after"], metrics_json["loss_before"])
            self.assertEqual(metrics_json["feature_count"], 2)
            self.assertEqual(metrics_json["example_count"], 4)

    def test_cli_train_local_supports_mlp_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tempdir = Path(temporary_directory)
            initialized_state = tempdir / "mlp-state.json"
            trained_state = tempdir / "trained-mlp-state.json"
            metrics = tempdir / "mlp-metrics.json"

            self._run(
                "init-state",
                "--runtime",
                "mlp-regression",
                "--features",
                "2",
                "--hidden-size",
                "4",
                "--seed",
                "7",
                "--weight-scale",
                "0.3",
                "-o",
                str(initialized_state),
            )
            self._run(
                "train-local",
                str(initialized_state),
                str(FIXTURES / "mlp_dataset_worker_a.json"),
                "--runtime",
                "mlp-regression",
                "--steps",
                "40",
                "--learning-rate",
                "0.05",
                "--batch-size",
                "2",
                "--metrics-out",
                str(metrics),
                "-o",
                str(trained_state),
            )

            trained_json = json.loads(trained_state.read_text(encoding="utf-8"))
            metrics_json = json.loads(metrics.read_text(encoding="utf-8"))

            self.assertIn("parameters", trained_json)
            self.assertEqual(metrics_json["runtime"], "mlp-regression")
            self.assertLess(metrics_json["loss_after"], metrics_json["loss_before"])
            self.assertEqual(metrics_json["feature_count"], 2)
            self.assertEqual(metrics_json["example_count"], 5)

    def test_cli_train_local_supports_numpy_backend_and_npz_state_io(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tempdir = Path(temporary_directory)
            initial_state = tempdir / "linear-state.npz"
            trained_state = tempdir / "trained-state.npz"
            trained_state_json = tempdir / "trained-state.json"
            metrics = tempdir / "metrics.json"

            self._run(
                "convert-state",
                str(FIXTURES / "linear_initial_state.json"),
                "-o",
                str(initial_state),
            )
            self._run(
                "train-local",
                str(initial_state),
                str(FIXTURES / "linear_dataset_worker_a.json"),
                "--backend",
                "numpy",
                "--steps",
                "30",
                "--learning-rate",
                "0.05",
                "--batch-size",
                "2",
                "--metrics-out",
                str(metrics),
                "-o",
                str(trained_state),
            )
            self._run(
                "convert-state",
                str(trained_state),
                "-o",
                str(trained_state_json),
            )

            trained_json = json.loads(trained_state_json.read_text(encoding="utf-8"))
            metrics_json = json.loads(metrics.read_text(encoding="utf-8"))

            self.assertIn("parameters", trained_json)
            self.assertEqual(metrics_json["backend"], "numpy")
            self.assertLess(metrics_json["loss_after"], metrics_json["loss_before"])

    def test_training_checkpoint_roundtrip_preserves_late_gradients(self) -> None:
        state = ModelState.from_path(FIXTURES / "linear_initial_state.json")
        checkpoint = TrainingCheckpoint(
            run_name="demo-run",
            worker_id="worker-a",
            relay_urls=("ws://127.0.0.1:8765",),
            next_round=1,
            current_state=state,
            momentum_state=None,
            rounds=(
                TrainingRoundSummary(
                    round_index=0,
                    model_hash_before="a" * 64,
                    model_hash_after="b" * 64,
                    local_loss_before=1.0,
                    local_loss_after_inner=0.5,
                    local_loss_after_outer=0.4,
                    collected_event_count=1,
                    known_workers=("worker-a",),
                    collected_workers=("worker-a",),
                    completion_reason="timeout",
                    published_gradient_event_id="c" * 64,
                    published_heartbeat_event_id="d" * 64,
                    published_checkpoint_event_id="",
                    configured_relays=("ws://127.0.0.1:8765",),
                    published_heartbeat_relays=("ws://127.0.0.1:8765",),
                    published_gradient_relays=("ws://127.0.0.1:8765",),
                    published_checkpoint_relays=(),
                    collected_from_relays=("ws://127.0.0.1:8765",),
                    failed_relays=(),
                ),
            ),
            late_gradients=(
                LateGradientRecord(
                    round_index=0,
                    worker_id="worker-b",
                    event_id="e" * 64,
                    created_at=1_700_000_000,
                    model_hash="f" * 64,
                    payload="payload-1",
                    reconciliation_round=1,
                    reconciliation_model_hash_before="1" * 64,
                    reconciliation_model_hash_after="2" * 64,
                ),
            ),
            late_reconciliations=(
                LateGradientReconciliationSummary(
                    current_round=1,
                    event_count=1,
                    worker_ids=("worker-b",),
                    late_rounds=(0,),
                    event_ids=("e" * 64,),
                    learning_rate=0.7,
                    momentum=0.9,
                    model_hash_before="1" * 64,
                    model_hash_after="2" * 64,
                    applied_at=1_700_000_050,
                ),
            ),
            updated_at=1_700_000_100,
            runtime_name="mlp-regression",
        )

        restored = TrainingCheckpoint.from_json_obj(checkpoint.to_json_obj())

        self.assertEqual(restored.next_round, 1)
        self.assertEqual(restored.rounds_completed, 1)
        self.assertEqual(len(restored.late_gradients), 1)
        self.assertEqual(restored.late_gradients[0].worker_id, "worker-b")
        self.assertEqual(restored.late_gradients[0].payload, "payload-1")
        self.assertEqual(restored.late_gradients[0].reconciliation_round, 1)
        self.assertEqual(restored.late_gradients[0].reconciliation_model_hash_after, "2" * 64)
        self.assertEqual(len(restored.late_reconciliations), 1)
        self.assertEqual(restored.late_reconciliations[0].event_count, 1)
        self.assertEqual(restored.late_reconciliations[0].worker_ids, ("worker-b",))
        self.assertEqual(restored.runtime_name, "mlp-regression")


if __name__ == "__main__":
    unittest.main()
