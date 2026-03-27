from __future__ import annotations

import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nostrain.model import ModelState

ROOT = Path(__file__).resolve().parents[1]
FAKE_TORCH_ROOT = ROOT / "tests" / "fake_torch"


def build_test_env(*, include_fake_torch: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(ROOT / "src")]
    if include_fake_torch:
        pythonpath_entries.append(str(FAKE_TORCH_ROOT))
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    return env


@contextmanager
def fake_torch_imports():
    fake_torch_path = str(FAKE_TORCH_ROOT)
    previous_module = sys.modules.pop("torch", None)
    sys.path.insert(0, fake_torch_path)
    try:
        yield
    finally:
        sys.modules.pop("torch", None)
        try:
            sys.path.remove(fake_torch_path)
        except ValueError:
            pass
        if previous_module is not None:
            sys.modules["torch"] = previous_module


def assert_state_json_almost_equal(
    test_case: unittest.TestCase,
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    places: int,
) -> None:
    left_parameters = left["parameters"]
    right_parameters = right["parameters"]
    test_case.assertEqual(sorted(left_parameters), sorted(right_parameters))

    for name in sorted(left_parameters):
        left_tensor = left_parameters[name]
        right_tensor = right_parameters[name]
        test_case.assertEqual(left_tensor["shape"], right_tensor["shape"])
        test_case.assertEqual(len(left_tensor["values"]), len(right_tensor["values"]))
        for left_value, right_value in zip(left_tensor["values"], right_tensor["values"]):
            test_case.assertAlmostEqual(left_value, right_value, places=places)


def assert_model_state_almost_equal(
    test_case: unittest.TestCase,
    left: ModelState,
    right: ModelState,
    *,
    places: int,
) -> None:
    assert_state_json_almost_equal(
        test_case,
        left.to_json_obj(),
        right.to_json_obj(),
        places=places,
    )
