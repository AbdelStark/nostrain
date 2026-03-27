from __future__ import annotations

import unittest
from pathlib import Path

from nostrain.aggregation import aggregate_deltas, nesterov_outer_step
from nostrain.model import ModelState, compute_delta
from tests.helpers import assert_model_state_almost_equal

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class AggregationTests(unittest.TestCase):
    def test_aggregate_deltas_averages_multiple_workers(self) -> None:
        initial = ModelState.from_path(FIXTURES / "initial_state.json")
        worker_a = ModelState.from_path(FIXTURES / "current_state.json")
        worker_b = ModelState.from_path(FIXTURES / "current_state_peer.json")

        delta_a = compute_delta(initial, worker_a)
        delta_b = compute_delta(initial, worker_b)
        aggregated = aggregate_deltas([delta_a, delta_b])
        expected = compute_delta(
            initial,
            ModelState.from_json_obj(
                {
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
            ),
        )

        assert_model_state_almost_equal(self, aggregated, expected, places=12)

    def test_outer_step_uses_velocity_and_returns_next_momentum(self) -> None:
        initial = ModelState.from_path(FIXTURES / "initial_state.json")
        worker_a = ModelState.from_path(FIXTURES / "current_state.json")
        worker_b = ModelState.from_path(FIXTURES / "current_state_peer.json")
        aggregated = aggregate_deltas(
            [
                compute_delta(initial, worker_a),
                compute_delta(initial, worker_b),
            ]
        )

        result = nesterov_outer_step(
            initial,
            aggregated,
            learning_rate=1.0,
            momentum=0.0,
        )
        expected_next = ModelState.from_json_obj(
            {
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
        )

        assert_model_state_almost_equal(self, result.next_state, expected_next, places=12)
        assert_model_state_almost_equal(self, result.momentum_state, aggregated, places=12)


if __name__ == "__main__":
    unittest.main()
