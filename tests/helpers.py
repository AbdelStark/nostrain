from __future__ import annotations

from typing import Any
import unittest

from nostrain.model import ModelState


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
