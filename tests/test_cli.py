from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import assert_state_json_almost_equal

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEST_SECRET_KEY = "0000000000000000000000000000000000000000000000000000000000000003"
TEST_PUBLIC_KEY = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"


class CliWorkflowTests(unittest.TestCase):
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

    def test_end_to_end_local_cli_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tempdir = Path(temporary_directory)
            payload_path = tempdir / "payload.json"
            payload_peer_path = tempdir / "payload-peer.json"
            reconstructed_path = tempdir / "reconstructed.json"
            aggregated_path = tempdir / "aggregated.json"
            event_path = tempdir / "event.json"
            heartbeat_path = tempdir / "heartbeat.json"
            summary_path = tempdir / "summary.json"
            heartbeat_summary_path = tempdir / "heartbeat-summary.json"
            next_state_path = tempdir / "next-state.json"
            momentum_path = tempdir / "momentum.json"

            digest = self._run("hash-state", str(FIXTURES / "initial_state.json")).stdout.strip()
            self.assertEqual(len(digest), 64)

            derived_pubkey = self._run("derive-pubkey", TEST_SECRET_KEY).stdout.strip()
            self.assertEqual(derived_pubkey, TEST_PUBLIC_KEY)

            self._run(
                "encode-delta",
                str(FIXTURES / "initial_state.json"),
                str(FIXTURES / "current_state.json"),
                "--topk",
                "1.0",
                "-o",
                str(payload_path),
            )
            payload_json = json.loads(payload_path.read_text(encoding="utf-8"))
            self.assertIn("payload", payload_json)
            self.assertEqual(payload_json["stats"]["selected_values"], 12)

            self._run(
                "encode-delta",
                str(FIXTURES / "initial_state.json"),
                str(FIXTURES / "current_state_peer.json"),
                "--topk",
                "1.0",
                "-o",
                str(payload_peer_path),
            )

            self._run(
                "apply-payload",
                str(FIXTURES / "initial_state.json"),
                str(payload_path),
                "-o",
                str(reconstructed_path),
            )
            reconstructed = json.loads(reconstructed_path.read_text(encoding="utf-8"))
            current = json.loads((FIXTURES / "current_state.json").read_text(encoding="utf-8"))
            assert_state_json_almost_equal(self, reconstructed, current, places=2)

            self._run(
                "aggregate-payloads",
                str(payload_path),
                str(payload_peer_path),
                "-o",
                str(aggregated_path),
            )
            aggregated = json.loads(aggregated_path.read_text(encoding="utf-8"))
            self.assertIn("parameters", aggregated)

            self._run(
                "outer-step",
                str(FIXTURES / "initial_state.json"),
                str(aggregated_path),
                "--learning-rate",
                "1.0",
                "--momentum",
                "0.0",
                "--momentum-out",
                str(momentum_path),
                "-o",
                str(next_state_path),
            )
            next_state = json.loads(next_state_path.read_text(encoding="utf-8"))
            expected_average = {
                "parameters": {
                    "encoder.bias": {
                        "shape": [3],
                        "values": [0.0115, -0.0225, 0.0325],
                    },
                    "encoder.weight": {
                        "shape": [2, 3],
                        "values": [0.17, -0.11, 0.295, 0.435, -0.535, 0.615],
                    },
                    "head.weight": {
                        "shape": [1, 3],
                        "values": [0.625, -0.25, 0.77],
                    },
                }
            }
            assert_state_json_almost_equal(self, next_state, expected_average, places=2)
            self.assertTrue(momentum_path.exists())

            self._run(
                "build-event",
                str(payload_path),
                "--run",
                "demo-run",
                "--round",
                "7",
                "--worker",
                "worker-pubkey",
                "--model",
                digest,
                "--steps",
                "500",
                "--examples",
                "11",
                "--created-at",
                "1700000000",
                "--sec-key",
                TEST_SECRET_KEY,
                "-o",
                str(event_path),
            )
            self.assertTrue(event_path.exists())

            self._run("inspect-event", str(event_path), "--json", "-o", str(summary_path))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["run"], "demo-run")
            self.assertEqual(summary["round"], 7)
            self.assertEqual(summary["worker"], "worker-pubkey")
            self.assertTrue(summary["signed"])
            self.assertEqual(summary["signing_state"], "signed")
            self.assertEqual(summary["pubkey"], TEST_PUBLIC_KEY)
            self.assertEqual(summary["example_count"], 11)
            self.assertEqual(summary["aggregation_weight"], 11)
            self.assertEqual(len(summary["event_id"]), 64)

            self._run(
                "build-heartbeat",
                "--run",
                "demo-run",
                "--round",
                "7",
                "--worker",
                "worker-pubkey",
                "--examples",
                "13",
                "--capability",
                "gradient-event",
                "--advertise-relay",
                "ws://127.0.0.1:8765",
                "--created-at",
                "1700000001",
                "--sec-key",
                TEST_SECRET_KEY,
                "-o",
                str(heartbeat_path),
            )
            self._run(
                "inspect-event",
                str(heartbeat_path),
                "--json",
                "-o",
                str(heartbeat_summary_path),
            )
            heartbeat_summary = json.loads(heartbeat_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(heartbeat_summary["type"], "heartbeat")
            self.assertEqual(heartbeat_summary["run"], "demo-run")
            self.assertEqual(heartbeat_summary["round"], 7)
            self.assertEqual(heartbeat_summary["worker"], "worker-pubkey")
            self.assertEqual(heartbeat_summary["heartbeat_interval"], 60)
            self.assertEqual(heartbeat_summary["example_count"], 13)
            self.assertEqual(heartbeat_summary["capabilities"], ["gradient-event"])
            self.assertEqual(
                heartbeat_summary["advertised_relays"],
                ["ws://127.0.0.1:8765"],
            )
            self.assertTrue(heartbeat_summary["signed"])


if __name__ == "__main__":
    unittest.main()
