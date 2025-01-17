# %%

from __future__ import annotations

import argparse
import logging
import pathlib
from collections import namedtuple
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from numpy.typing import NDArray

LogisticParams = namedtuple("LogisticParams", ["scale", "coef", "intercept"])


def unscaled_regression(
    X: NDArray[Any],
    y: NDArray[Any],
    sample_weight: NDArray[Any],
    regularization: float = 0.01,
) -> LogisticRegression:
    # Assert y values are in [0,1]
    assert np.all((y >= 0) & (y <= 1)), "y values must be in [0,1]"
    original_weight_sum = np.sum(sample_weight)
    original_average = np.average(y, weights=sample_weight)

    # For any fractional y values, split into weighted 0s and 1s
    fractional_mask = (y > 0) & (y < 1)
    if np.any(fractional_mask):
        X_frac = X[fractional_mask]
        y_frac = y[fractional_mask]
        w_frac = sample_weight[fractional_mask]

        # Stack X twice for 0s and 1s
        X_split = np.vstack([X_frac, X_frac])

        # Create y array with 0s in first half, 1s in second half
        y_split = np.zeros(2 * len(y_frac))
        y_split[len(y_frac) :] = 1

        # Weight the 0s by (1-y) and 1s by y
        w_split = np.concatenate([(1 - y_frac) * w_frac, y_frac * w_frac])

        # Combine with non-fractional values
        X = np.vstack([X[~fractional_mask], X_split])
        y = np.concatenate([y[~fractional_mask], y_split])
        sample_weight = np.concatenate([sample_weight[~fractional_mask], w_split])
        assert np.allclose(np.sum(sample_weight), original_weight_sum)
        assert np.allclose(np.average(y, weights=sample_weight), original_average)

    model = LogisticRegression(C=1 / regularization)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def get_x_for_quantile(model: LogisticRegression, quantile: float) -> float:
    return (np.log(quantile / (1 - quantile)) - model.intercept_[0]) / model.coef_[0][0]  # type: ignore


def empirical_success_rates(
    x: NDArray[Any],
    y: NDArray[Any],
    time_buckets: list[int],
    weights: NDArray[Any],
) -> tuple[pd.Series[Any], float]:
    use_weighted_mean = True
    # Calculate empirical success rates for different time buckets
    empirical_rates = []
    for i in range(len(time_buckets) - 1):
        mask = (np.exp2(x).reshape(-1) >= time_buckets[i]) & (
            np.exp2(x).reshape(-1) < time_buckets[i + 1]
        )
        success_rate = (
            np.sum(y[mask] * weights[mask]) / np.sum(weights[mask])
            if use_weighted_mean
            else np.mean(y[mask])
        )
        empirical_rates.append(success_rate)

    average = np.sum(y * weights) / np.sum(weights)
    indices = [
        f"{start}-{end} min" for start, end in zip(time_buckets[:-1], time_buckets[1:])
    ]
    return pd.Series(empirical_rates, index=indices), average


def get_accuracy(
    x: NDArray[Any],
    y: NDArray[Any],
    model: LogisticRegression,
) -> float:
    return model.score(x, y)


def get_bce_loss(
    x: NDArray[Any],
    y: NDArray[Any],
    model: LogisticRegression,
    weights: NDArray[Any],
) -> float:
    y_pred = model.predict_proba(x)[:, 1]

    # Calculate weighted BCE loss
    # can't use sklearn.metrics.log_loss because it doesn't support continuous y
    epsilon = 1e-15  # small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    weights = weights / weights.mean()
    bce = -weights * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return np.mean(bce).item()


def agent_regression(
    x: NDArray[Any],
    y: NDArray[Any],
    weights: NDArray[Any] | None,
    agent_name: str,
    method: str,
    regularization: float = 0.01,
) -> pd.Series[Any]:
    if method != "unscaled":
        raise ValueError(f"Unknown method: {method}")
    regression = unscaled_regression

    logging.info(f"Analyzing {agent_name}, method {method}")
    time_buckets = [1, 4, 16, 64, 256, 960]
    y = np.clip(y, 0, 1)
    x = np.log2(x).reshape(-1, 1)
    if weights is None:
        weights = np.ones_like(y, dtype=np.float64)

    empirical_rates, average = empirical_success_rates(x, y, time_buckets, weights)

    indices = [
        "scale",
        "coefficient",
        "intercept",
        "bce_loss",
        "20%",
        "50%",
        "50_full",
        "50_low",
        "50_high",
        "average",
    ]
    if np.all(y == 0):
        return pd.Series(
            [
                0,
                -np.inf,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            index=indices,
        )._append(empirical_rates)  # type: ignore[reportCallIssue]
    # get confidence interval for p50. TODO remove this, should be using bootstrap
    kf = KFold(n_splits=min(10, len(x)), shuffle=True, random_state=42)
    p50s = []
    for train_idx, _ in kf.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        params = regression(
            x_train,
            y_train,
            sample_weight=weights[train_idx],
            regularization=regularization,
        )
        p50_boot = np.exp2(get_x_for_quantile(params, 0.5))
        p50s.append(p50_boot)
    p50 = np.percentile(p50s, 50)
    p50low = np.percentile(p50s, 2.5)
    p50high = np.percentile(p50s, 97.5)

    model = regression(x, y, sample_weight=weights, regularization=regularization)
    if model.coef_[0][0] > 0:
        logging.warning(f"Warning: {agent_name} has positive slope {model.coef_[0][0]}")
    p50_full = np.exp2(get_x_for_quantile(model, 0.5))
    p20 = np.exp2(get_x_for_quantile(model, 0.2))

    bce_loss = get_bce_loss(x, y, model, weights)
    print(type(bce_loss))

    return pd.Series(
        [
            1,
            model.coef_[0][0],
            model.intercept_[0],  # type: ignore
            bce_loss,
            p20,
            p50,
            p50_full,
            p50low,
            p50high,
            average,
        ],
        index=indices,
    )._append(empirical_rates)


def run_logistic_regression(
    runs: pd.DataFrame,
    output_file: pathlib.Path,
    weighting: str,
    method: str,
    regularization: float = 0.01,
) -> None:
    weights_fn = lambda x: None if weighting == "None" else x[weighting].values  # noqa: E731
    # rename alias to agent
    runs.rename(columns={"alias": "agent"}, inplace=True)
    regressions = runs.groupby(["agent"]).apply(
        lambda x: agent_regression(
            x["human_minutes"].values,  # type: ignore
            x["score"].values,  # type: ignore
            weights=weights_fn(x),  # type: ignore
            agent_name=x.name,  # type: ignore
            method=method,
            regularization=regularization,
        )  # type: ignore
    )  # type: ignore
    # ungroup and turn into DataFrame

    logging.info(regressions)
    logging.info(f"Mean BCE loss: {regressions.bce_loss.mean():.3f}")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    regressions.to_csv(output_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--weighting", type=str, required=True)
    parser.add_argument("--method", type=str, default="unscaled")
    parser.add_argument("--regularization", type=float, default=0.01)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    runs = pd.read_json(
        args.input_file, lines=True, orient="records", convert_dates=False
    )
    logging.info("Loaded input data")

    run_logistic_regression(
        runs, args.output_file, args.weighting, args.method, args.regularization
    )


if __name__ == "__main__":
    main()

# %%
