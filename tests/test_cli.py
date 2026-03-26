from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

from tests.helpers import assert_state_json_almost_equal


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


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
            reconstructed_path = tempdir / "reconstructed.json"
            event_path = tempdir / "event.json"
            summary_path = tempdir / "summary.json"

            digest = self._run("hash-state", str(FIXTURES / "initial_state.json")).stdout.strip()
            self.assertEqual(len(digest), 64)

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
                "--created-at",
                "1700000000",
                "-o",
                str(event_path),
            )
            self.assertTrue(event_path.exists())

            self._run("inspect-event", str(event_path), "--json", "-o", str(summary_path))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["run"], "demo-run")
            self.assertEqual(summary["round"], 7)
            self.assertEqual(summary["worker"], "worker-pubkey")


if __name__ == "__main__":
    unittest.main()
