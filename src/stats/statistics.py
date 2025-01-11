from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_best_of_k_weights(n: int, k: int) -> NDArray[np.float64]:
    """Compute numerically stable weights for best-of-k calculation.

    Args:
        n: Number of samples
        k: Number of independent draws to take maximum over

    Returns:
        Array of weights for sorted scores

    Note:
        Taken from https://arxiv.org/pdf/2406.06647 (Algorithm 1)
    """
    # Create array of r values from n-1 down to k-1
    r_values = np.arange(n - 1, k - 1, -1)

    # Initialize with first weight
    lambda_r = k / n

    # Calculate all subsequent weights
    factors = 1 - (k - 1) / r_values
    weights = lambda_r * np.cumprod(factors)

    # Prepend the initial lambda_r and pad with zeros
    weights = np.concatenate([[lambda_r], weights])
    return np.pad(weights, (0, n - len(weights)), constant_values=0)


def get_score_at_k_from_sample_deterministic(
    scores: NDArray[np.float64],
    ks: list[int],
    task_id: str | None = None,
    drop_too_few_scores: bool = False,
    print_warning: bool = False,
) -> NDArray[np.float64]:
    """Compute the mean over the max scores of all the n choose k groups.

    Args:
        scores: Array of scores to compute best-of-k over
        ks: List of k values to compute best-of-k for
        task_id: Identifier for the task
        drop_too_few_scores: If True, return nan when k > n. If False, use k = n

    Returns:
        Array of best-of-k scores for each k value

    Note:
        Taken from https://arxiv.org/pdf/2406.06647 (Algorithm 1)
        Computes the mean over the max scores of all the n choose k groups.
        Works for both binary and continuous scoring.
    """
    n = len(scores)

    # TODO: verify whether this is appropriate for aggregating BETWEEN run scores, rather than just the within run scores
    if task_id == "ai_rd_small_scaling_law/main":
        return np.full(len(ks), np.mean(scores)).astype(np.float64)

    # Sort scores in descending order once
    sorted_scores = np.sort(scores)[::-1]

    # Handle k > n cases
    original_ks = np.array(ks)
    new_ks = np.array(ks)
    if drop_too_few_scores:
        results = np.full(len(new_ks), np.nan)
        valid_k_mask = new_ks <= n
        new_ks = np.where(valid_k_mask, new_ks, n)
        invalid_ks = original_ks[~valid_k_mask]
        if print_warning and len(invalid_ks) > 0:
            logging.warning(
                f"Warning: The following k values cannot be filled: {invalid_ks} for the task, returning nan"
            )
    else:
        results = np.zeros(len(new_ks))
        new_ks = np.minimum(new_ks, n)

    # Compute weights for all k values at once
    weights_matrix = np.array([compute_best_of_k_weights(n, k) for k in new_ks])

    # Normalize weights
    weights_sums = weights_matrix.sum(axis=1, keepdims=True)
    weights_matrix = weights_matrix / weights_sums

    # Compute weighted sums for all k values at once
    results = np.where(
        original_ks <= n if drop_too_few_scores else True,
        np.sum(weights_matrix * sorted_scores, axis=1),
        np.nan,
    )

    return results


def get_bootstrapped_score_at_k(
    scores: NDArray[np.float64],
    ks: list[int],
    task_id: str,
    n_bootstrap: int = 5_000,
) -> NDArray[np.float64]:
    """Uses bootstrapping to generate a list of point estimates for the mean of the distribution of best-of-k for a task,
    from the real observed scores on the task.
    Uses the same bootstrap samples to generate an estimate for each k value.
    (except for small_scaling_law, where we take the mean instead of best-of-k)

    Args:
        scores: Array of scores for the task
        ks: List of k values to take best-of-k or mean-of-k over
        task_id: Identifier for the task
        n_bootstrap: Number of bootstrap iterations
    Returns:
        Array of point estimates of the mean of the best-of-k distribution. shape [n_bootstrap, len(ks)]
    """
    if len(scores) == 0:
        raise ValueError("No scores provided")
    if np.any(np.isnan(scores)):
        raise ValueError("Scores contain NaN values")

    scores = np.asarray(scores)
    sample_size = len(scores)

    # Generate all bootstrap samples at once
    rng = np.random.default_rng()
    sampled_scores = rng.choice(scores, size=(n_bootstrap, sample_size), replace=True)

    # Process all bootstrap samples at once
    bootstrap_means = np.array(
        [
            get_score_at_k_from_sample_deterministic(sample, ks, task_id)
            for sample in sampled_scores
        ]
    )

    return bootstrap_means


def get_cross_task_summary_statistics(
    scores_by_task: list[NDArray[np.float64]],
    task_ids: list[str],
    k: int,
    n_bootstrap: int = 5_000,
) -> tuple[float, float, float, float]:
    """Get summary statistics averaged across tasks.

    Returns:
        Tuple of (point_estimate, mean_bootstrap, ci_lower, ci_upper)
    """
    # Compute point estimates for all tasks at once
    point_estimates = np.array(
        [
            get_score_at_k_from_sample_deterministic(scores, [k], task_id)[0]
            for scores, task_id in zip(scores_by_task, task_ids)
        ]
    )

    # Get bootstrap samples for all tasks at once
    processed_per_task = np.array(
        [
            get_bootstrapped_score_at_k(scores, [k], task_id, n_bootstrap)[:, 0]
            for scores, task_id in zip(scores_by_task, task_ids)
        ]
    )

    # Average across tasks
    processed = processed_per_task.mean(axis=0)

    # Compute confidence intervals
    ci_lower, ci_upper = np.percentile(processed, [2.5, 97.5])

    return np.mean(point_estimates).astype(float), processed.mean(), ci_lower, ci_upper
