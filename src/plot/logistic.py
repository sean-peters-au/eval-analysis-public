from __future__ import annotations

import argparse
import logging
import os
import pathlib
from typing import Any, Sequence, cast

import dvc.api
import matplotlib.axes
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.optimize import Bounds, curve_fit
from sklearn.linear_model import LinearRegression

import src.utils.plots


def plot_logistic_regression(
    plot_params: src.utils.plots.PlotParams,
    agent_summaries: pd.DataFrame,
    output_file: pathlib.Path,
    focus_agents: Sequence[str],
    trendlines: bool = True,
    after_date: str = "2022-06-01",
    subtitle: str = "Sample Weight",
) -> None:
    agent_summaries = agent_summaries[
        agent_summaries["agent"].isin(focus_agents)
    ].copy()
    fig, ax = plt.subplots()
    agent_summaries["50%_clipped"] = agent_summaries["50%"].clip(
        # np.finfo(float).eps, np.inf
        0.5,
        np.inf,
    )  # clip because log scale makes 0 -> -inf
    agent_summaries.loc[:, "50_low"].clip(1, np.inf)
    agent_summaries.loc[:, "50_high"].clip(1, np.inf)
    ax.errorbar(
        agent_summaries["release_date"],
        agent_summaries["50%_clipped"],
        yerr=(agent_summaries["50_high"] - agent_summaries["50_low"]) / 2,
        fmt="o",
        markersize=0,  # invisible
        capsize=5,
        ecolor="gray",
    )

    # print(agent_summaries["50%"])
    for i, agent in enumerate(agent_summaries["agent"]):
        agent_color = src.utils.plots.get_agent_color(plot_params["colors"], agent)
        ax.scatter(
            agent_summaries["release_date"].iloc[i],
            agent_summaries["50%_clipped"].iloc[i],
            marker="o",
            edgecolor=agent_color if agent_summaries["50%"].iloc[i] > 0.5 else "black",
            facecolor=agent_color if agent_summaries["50%"].iloc[i] > 0.5 else "white",
            label=agent,
        )

    # Add arrows for out-of-range points
    mask_out_range = agent_summaries["50%_clipped"] != agent_summaries["50%"]
    ax.scatter(
        agent_summaries.loc[mask_out_range, "release_date"],
        [0.4] * mask_out_range.sum(),  # Place at bottom of visible range
        marker="v",  # Downward pointing triangle
        color="black",
    )

    # # Add agent labels to each point
    # texts = []
    # for _idx, row in agent_summaries.iterrows():
    #     texts.append(
    #         ax.text(
    #             row["release_date"] + pd.Timedelta(days=3),  # Offset text to the right
    #             row["50%_clipped"],  # Offset text slightly above point
    #             row["alias"],
    #         )
    #     )

    if trendlines:
        plot_trendline(ax, agent_summaries, after=after_date, log_scale=True)
        if after_date != "2024-01-01":
            plot_trendline(ax, agent_summaries, after=after_date, log_scale=False)
            plot_trendline_hyperbolic(ax, agent_summaries, after=after_date)
    # Adjust text positions to avoid overlaps
    # adjust_text(
    #     texts,
    #     max_move=7,
    # )

    if after_date == "2024-01-01":
        ax.set_xbound(
            pd.Timestamp("2023-01-01"),  # type: ignore[reportArgumentType]
            pd.Timestamp("2025-03-01"),  # type: ignore[reportArgumentType]
        )
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8, 15, 30, 60, 120])
    ax.set_yticklabels(
        ["1 min", "2 min", "4 min", "8 min", "15 min", "30 min", "1 hr", "2 hrs"]
    )
    ax.set_xlabel("Release Date")
    # ax.set_xticks(
    #     [
    #         pd.Timestamp(after_date) + pd.Timedelta(weeks=- 52 + 13*n)
    #         for n in range(0, 5)
    #     ]
    # )

    ax.set_ylabel("Human time-to-complete @ 50% chance of AI success")
    ax.set_title(f"Time horizon on well-defined tasks\n{subtitle}")

    # The graph is too busy if we have both trendlines and legend
    if not trendlines:
        src.utils.plots.create_sorted_legend(ax, plot_params["legend_order"])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file)
    logging.info(f"Saved plot to {output_file}")


def plot_trendline(
    ax: matplotlib.axes.Axes,
    agent_summaries: pd.DataFrame,
    after: str,
    log_scale: bool = False,
    method: str = "OLS",
    annotate: bool = True,
    fit_color: str | None = None,
) -> float | None:
    """Plot a trendline showing the relationship between release date and time horizon."""
    agent_summaries = agent_summaries.sort_values("release_date")
    mask = agent_summaries["release_date"] >= pd.Timestamp(after).date()
    # Convert release dates to numeric (days since first release)
    dates = pd.to_datetime(agent_summaries["release_date"])
    first_release = min(dates)
    days_since_release = (dates - first_release).dt.days

    # Fit linear regression
    X = (cast(NDArray[Any], days_since_release.values)).reshape(-1, 1)
    y_raw = agent_summaries["50%"].clip(np.finfo(float).eps.astype(float), np.inf)
    # clip
    y_raw = y_raw.clip(0.5, np.inf)
    y = np.log(y_raw) if log_scale else y_raw

    X = X[mask]
    y = y[mask]
    if method == "WLS":
        spread = np.log(agent_summaries["50_high"]) - np.log(agent_summaries["50_low"])
        weights = 1 / (spread**2 + 1e-6)
        # replace nan with 0
        weights = np.nan_to_num(weights)[mask]
    else:
        weights = None
    reg = LinearRegression().fit(X, y, sample_weight=weights)

    # Generate points for trendline, 20 equally spaced
    # x_range = np.array([min(X)[0], max(X)[0]])
    x_range = np.linspace(min(X)[0], max(X)[0], 20)
    y_pred = reg.predict(x_range.reshape(-1, 1))

    # Convert back to datetime for plotting
    x_dates = [first_release + pd.Timedelta(days=x) for x in x_range]
    y_values = np.exp(y_pred) if log_scale else y_pred  # Convert back from log scale

    # print(f"x_dates: {x_dates}")
    # print(f"y_values: {y_values}")

    fit_color = fit_color or ("blue" if log_scale else "red")
    # Plot trendline
    ax.plot(x_dates, y_values, "--", color=fit_color, alpha=0.5)  # type: ignore[reportArgumentType]

    doubling_time = 1 / reg.coef_[0] * np.log(2) if log_scale else None
    if not annotate:
        return doubling_time

    fit_type = "Exponential" if log_scale else "Linear"
    annotation = f"{fit_type} fit\nr^2: {reg.score(X, y):.2f}"
    if log_scale:
        annotation += f"\n(Doubling time: {doubling_time:.0f} days)"
        annotation += "\n" + ("All data" if after == "0000-00-00" else f"{after}+ data")
    else:
        annotation += f"\n(Rate: +{reg.coef_[0]*365:.0f} minutes per year)"

    annotation_x = (
        pd.Timestamp("2023-03-01") if after == "2024-01-01" else ax.get_xlim()[0] + 50
    )
    ax.annotate(
        annotation,
        (
            annotation_x,  # type: ignore[reportArgumentType]
            20 if log_scale else 8,
        ),
        xytext=(0, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=fit_color,
    )
    return doubling_time


def plot_trendline_hyperbolic(
    ax: matplotlib.axes.Axes, agent_summaries: pd.DataFrame, after: str
) -> None:
    """Plot a trendline showing the relationship between release date and time horizon."""

    def hyperbolic_func(
        x: NDArray[Any], a: NDArray[Any], t_agi: NDArray[Any]
    ) -> NDArray[Any]:
        """
        Tries to fit the y value in log space
        """
        return np.log(a / (t_agi - x))

    log_scale = True
    agent_summaries = agent_summaries.sort_values("release_date")
    mask = agent_summaries["release_date"] >= pd.Timestamp(after).date()
    # Convert release dates to numeric (days since first release)
    dates = pd.to_datetime(agent_summaries["release_date"])
    first_release = min(dates)
    days_since_release = (dates - first_release).dt.days

    # Fit linear regression
    X = (cast(NDArray[Any], days_since_release.values)).reshape(-1, 1)
    y_raw = agent_summaries["50%"].clip(np.finfo(float).eps.astype(float), np.inf)
    y = np.log(y_raw) if log_scale else y_raw
    X = X[mask]
    y = y[mask]

    bounds = Bounds([0, max(X)[0]], [np.inf, np.inf])  # type: ignore[reportArgumentType]
    params, _ = curve_fit(
        hyperbolic_func, X.flatten(), y, bounds=bounds, p0=[5, 365 * 5.0]
    )
    # print(f"params: {params}")

    # get r^2
    residuals = y - hyperbolic_func(X.flatten(), *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # type: ignore[reportOperatorIssue]
    r_squared = 1 - (ss_res / ss_tot)

    # Generate points for trendline, 20 equally spaced
    # x_range = np.array([min(X)[0], max(X)[0]])
    x_range = np.linspace(min(X)[0], max(X)[0], 20)
    y_pred = hyperbolic_func(x_range, params[0], params[1])

    # Convert back to datetime for plotting
    x_dates = [first_release + pd.Timedelta(days=x) for x in x_range]
    y_values = np.exp(y_pred) if log_scale else y_pred  # Convert back from log scale

    # print(f"x_dates: {x_dates}")
    # print(f"y_values: {y_values}")

    fit_color = "green"

    # Plot trendline
    ax.plot(x_dates, y_values, "--", color=fit_color, alpha=0.5)  # type: ignore[reportArgumentType]
    fit_type = "Hyperbolic"
    annotation = f"{fit_type} fit\nr^2: {r_squared:.2f}"
    t_agi = params[1]
    # convert to date
    t_agi = first_release + pd.Timedelta(days=t_agi)
    annotation += f"\n(T_agi: {t_agi:%Y-%m-%d})"

    ax.annotate(
        annotation,
        (first_release, 3),  # type: ignore[reportArgumentType]
        xytext=(0, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=fit_color,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--plot-format", type=str, default="png")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--trendlines", type=str, default="true")
    parser.add_argument("--after-date", type=str, default="2022-06-01")
    parser.add_argument("--subtitle", type=str, default="Sample Weight")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    agent_summaries = pd.read_csv(args.input_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())
    agent_summaries["release_date"] = agent_summaries["agent"].map(
        release_dates["date"]
    )
    focus_agents = [
        "Claude 3 Opus",
        "Claude 3.5 Sonnet (New)",
        "Claude 3.5 Sonnet (Old)",
        "GPT-4 0314",
        "GPT-4 Turbo",
        "GPT-4o",
        "davinci-002",
        "gpt-3.5-turbo-instruct",
        "o1",
        "o1-preview",
    ]

    logging.info("Loaded input data")
    trendlines = args.trendlines.lower() == "true"

    params = dvc.api.params_show(stages="plot_logistic_regression")

    plot_logistic_regression(
        params["plots"],
        agent_summaries,
        args.output_file,
        focus_agents,
        trendlines=trendlines,
        after_date=args.after_date,
        subtitle=args.subtitle,
    )


if __name__ == "__main__":
    main()
