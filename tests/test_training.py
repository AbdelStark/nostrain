from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nostrain.model import ModelState
from nostrain.training import (
    LinearRegressionDataset,
    LocalTrainingConfig,
    evaluate_linear_regression,
    train_linear_regression,
)


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


if __name__ == "__main__":
    unittest.main()
