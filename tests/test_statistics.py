from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import src.stats.statistics as statistics  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.mark.parametrize(
    ("scores", "n_samples", "task_id", "n_bootstrap", "expected_properties"),
    [
        # Single value case
        (
            np.array([0.5]),
            1,
            "ai_rd_small_scaling_law/main",
            100,
            {"mean": 0.5, "std": 0.0},
        ),
        # Basic case for small_scaling_law (should use mean)
        (
            np.array([0.0, 1.0]),
            2,
            "ai_rd_small_scaling_law/main",
            1000,
            {"mean": lambda x: abs(x - 0.5) < 0.1, "std": lambda x: x > 0},
        ),
        # Basic case for other tasks (should use max)
        (
            np.array([0.0, 1.0]),
            2,
            "other_task",
            1000,
            {"mean": lambda x: abs(x - 0.75) < 0.1, "std": lambda x: x > 0},
        ),
        # Test with n_samples < len(scores)
        (
            np.array([0.0, 0.5, 1.0]),
            2,
            "ai_rd_small_scaling_law/main",
            1000,
            {"mean": lambda x: abs(x - 0.5) < 0.1, "std": lambda x: x > 0},
        ),
    ],
)
def test_bootstrapped_score_at_k(
    scores: NDArray[Any],
    n_samples: int,
    task_id: str,
    n_bootstrap: int,
    expected_properties: dict[str, Any],
) -> None:
    result = statistics.get_bootstrapped_score_at_k(
        scores, [n_samples], task_id, n_bootstrap
    ).flatten()

    assert len(result) == n_bootstrap
    assert not np.any(np.isnan(result))

    for prop, expected in expected_properties.items():
        actual = getattr(np, prop)(result)
        if callable(expected):
            assert expected(
                actual
            ), f"Expected {prop} to satisfy condition, got {actual}"
        else:
            # Only use exact equality for edge cases like std=0 with single value
            assert actual == expected, f"Expected {prop} to be {expected}, got {actual}"


@pytest.mark.parametrize(
    ("scores", "n_samples", "task_id", "n_bootstrap", "expected_properties"),
    [
        # Single value case
        (
            np.array([0.5]),
            1,
            "ai_rd_small_scaling_law/main",
            100,
            {"mean": 0.5, "std": 0.0},
        ),
        # Basic case for small_scaling_law (should use mean)
        (
            np.array([0.0, 1.0]),
            2,
            "ai_rd_small_scaling_law/main",
            1000,
            {"mean": lambda x: abs(x - 0.5) < 0.1, "std": lambda x: x > 0},
        ),
        # Basic case for other tasks (should use max)
        (
            np.array([0.0, 1.0]),
            2,
            "other_task",
            1000,
            {"mean": lambda x: abs(x - 0.75) < 0.1, "std": lambda x: x > 0},
        ),
        # Test with n_samples < len(scores)
        (
            np.array([0.0, 0.5, 1.0]),
            2,
            "ai_rd_small_scaling_law/main",
            1000,
            {"mean": lambda x: abs(x - 0.5) < 0.1, "std": lambda x: x > 0},
        ),
        # k value larger than the number of scores
        (
            np.array([0.0, 1.0]),
            3,
            "other_task",
            1000,
            {"mean": lambda x: abs(x - 0.7) < 0.1, "std": lambda x: x > 0},
        ),
    ],
)
def test_bootstrapped_score_at_k_deterministic(
    scores: NDArray[Any],
    n_samples: int,
    task_id: str,
    n_bootstrap: int,
    expected_properties: dict[str, Any],
) -> None:
    # drop_too_few_scores = len(scores) < n_samples
    result = statistics.get_bootstrapped_score_at_k(
        scores,
        [n_samples],
        task_id,
        n_bootstrap,
    ).flatten()

    assert len(result) == n_bootstrap
    assert not np.any(np.isnan(result))

    for prop, expected in expected_properties.items():
        actual = getattr(np, prop)(result)
        if callable(expected):
            assert expected(
                actual
            ), f"Expected {prop} to satisfy condition, got {actual}"
        else:
            # Only use exact equality for edge cases like std=0 with single value
            assert actual == expected, f"Expected {prop} to be {expected}, got {actual}"


@pytest.mark.parametrize(
    ("scores", "n_samples", "task_id", "expected_error"),
    [
        (np.array([]), 1, "any_task", "No scores provided"),  # Empty scores
        (
            np.array([1.0, np.nan]),
            1,
            "any_task",
            "Scores contain NaN values",
        ),  # Contains NaN
    ],
)
def test_bootstrapped_score_at_k_invalid_inputs(
    scores: NDArray[Any], n_samples: int, task_id: str, expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        statistics.get_bootstrapped_score_at_k(scores, [n_samples], task_id).flatten()
