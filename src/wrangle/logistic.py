# %%

from __future__ import annotations

import argparse
import logging
import pathlib
from collections import namedtuple
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from numpy.typing import NDArray

LogisticParams = namedtuple("LogisticParams", ["scale", "coef", "intercept"])


class ScaledLogistic(nn.Module):
    """
    Represents logistic regression with a scaling parameter
    y = alpha * sigmoid(theta^T x + b)
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # We'll treat theta and a as learnable parameters.
        # a is unconstrained in (-∞, +∞), so alpha = sigmoid(a) will be in [0,1].
        self.coef = nn.Parameter(
            -0.5 + 0.5 * torch.rand(input_dim, dtype=torch.float64)
        )
        self.a = nn.Parameter(
            torch.zeros(1, dtype=torch.float64)
        )  # initialize a at 0 => alpha=0.5
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float64))  # initialize b at 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.a)
        logits = torch.matmul(x, self.coef) + self.b
        p_scaled = scale * torch.sigmoid(logits)
        return p_scaled

    def train(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor,
        num_epochs: int = 5000,
    ) -> LogisticParams:
        sample_weight = sample_weight / sample_weight.mean()
        criterion = nn.BCELoss(
            weight=sample_weight
        )  # binary cross-entropy for probabilities
        optimizer = optim.Adam(self.parameters(), lr=0.01)

        # Training loop
        loss = None
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # p_hat in [0,1]
            p_hat = self(X)
            loss = criterion(p_hat, y)
            loss.backward()
            optimizer.step()

            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

        assert (
            loss is not None and (loss < 0.8)
        ), f"Something is wrong, loss is {loss:.2f}, worse than chance ({np.log(2):.2f})"

        # Final parameters
        with torch.no_grad():
            scale = torch.sigmoid(self.a).item()
            return LogisticParams(
                scale=scale, coef=self.coef.item(), intercept=self.b.item()
            )

    @staticmethod
    def get_x_for_quantile(
        params: LogisticParams, quantile: float = 0.5
    ) -> NDArray[Any]:
        coef = params.coef
        intercept = params.intercept
        quantile = quantile / params.scale
        ret = (np.log(quantile / (1 - quantile)) - intercept) / coef
        return ret.item()

    @staticmethod
    def regression(
        X: NDArray[Any], y: NDArray[Any], sample_weight: NDArray[Any]
    ) -> LogisticParams:
        model = ScaledLogistic(1)
        params = model.train(
            torch.tensor(X, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64),
            sample_weight=torch.tensor(sample_weight),
        )
        return params


# %%


def unscaled_regression(
    X: NDArray[Any], y: NDArray[Any], sample_weight: NDArray[Any]
) -> LogisticParams:
    model = LogisticRegression()
    model.fit(X, y, sample_weight=sample_weight)
    return LogisticParams(
        scale=1,
        coef=model.coef_[0][0],
        intercept=model.intercept_[0],  # type: ignore[reportIndexIssue]
    )


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


def get_accuracy(x: NDArray[Any], y: NDArray[Any], params: LogisticParams) -> float:
    y_pred = params.scale * torch.sigmoid(
        torch.matmul(
            torch.tensor(x, dtype=torch.float64),
            torch.tensor([params.coef], dtype=torch.float64),
        )
        + torch.tensor(params.intercept)
    )
    y_pred = y_pred.detach().numpy()
    return np.mean((y_pred > 0.5) == y).astype(float)


def agent_regression(
    x: NDArray[Any],
    y: NDArray[Any],
    weights: NDArray[Any],
    agent_name: str,
    method: str,
) -> pd.Series[Any]:
    if method == "scaled":
        regression = ScaledLogistic.regression
    else:
        regression = unscaled_regression

    print(f"Analyzing {agent_name}, method {method}")
    time_buckets = [1, 4, 16, 64, 256, 960]
    y = np.where(y > 0.9, 1, 0)
    x = np.log2(x).reshape(-1, 1)
    if weights is None:
        weights = np.ones_like(y, dtype=np.float64)

    # print(x, y)
    empirical_rates, average = empirical_success_rates(x, y, time_buckets, weights)

    indices = [
        "scale",
        "coefficient",
        "intercept",
        "r2",
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
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            index=indices,
        )._append(empirical_rates)  # type: ignore[reportCallIssue]
    # Bootstrap to get confidence interval for p50
    kf = KFold(n_splits=min(10, len(x)), shuffle=True, random_state=42)
    p50s = []
    for train_idx, _ in kf.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        params = regression(x_train, y_train, sample_weight=weights[train_idx])
        p50_boot = np.exp2(ScaledLogistic.get_x_for_quantile(params, 0.5))
        p50s.append(p50_boot)
    p50 = np.percentile(p50s, 50)
    p50low = np.percentile(p50s, 2.5)
    p50high = np.percentile(p50s, 97.5)

    params = regression(x, y, sample_weight=weights)
    if params.coef > 0:
        print(f"Warning: {agent_name} has positive slope {params.coef}")
    p50_full = np.exp2(ScaledLogistic.get_x_for_quantile(params, 0.5))
    p20 = np.exp2(ScaledLogistic.get_x_for_quantile(params, 0.2))

    accuracy = get_accuracy(x, y, params)

    return pd.Series(
        [
            params.scale,
            params.coef,
            params.intercept,
            accuracy,
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
    runs: pd.DataFrame, output_file: pathlib.Path, weighting: str, method: str
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
        )  # type: ignore
    )  # type: ignore
    # ungroup and turn into DataFrame

    print(regressions)
    regressions.to_csv(output_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--weighting", type=str, required=True)
    parser.add_argument("--method", type=str, default="unscaled")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    runs = pd.read_json(
        args.input_file, lines=True, orient="records", convert_dates=False
    )
    logging.info("Loaded input data")

    run_logistic_regression(runs, args.output_file, args.weighting, args.method)


if __name__ == "__main__":
    main()
