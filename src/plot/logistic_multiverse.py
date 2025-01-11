import argparse
import datetime
import logging
import pathlib
from typing import Sequence

import dvc.api
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import yaml
from adjustText import adjust_text
from matplotlib import pyplot as plt

import src.utils.plots
from src.plot.logistic import plot_trendline


def plot_multiverse(
    plot_params: src.utils.plots.PlotParams,
    ax: matplotlib.axes.Axes,
    agent_summaries: pd.DataFrame,
    focus_agents: Sequence[str],
    weighting: str,
) -> list[float | None]:
    # exclude agents with nan 50%
    agent_summaries = agent_summaries[agent_summaries["50%"].notna()]
    logging.info(
        f"Excluded {len(agent_summaries) - len(agent_summaries[agent_summaries['50%'].notna()])} agents with nan 50%"
    )
    agent_summaries = agent_summaries[agent_summaries["agent"].isin(focus_agents)]
    print(agent_summaries)
    agent_summaries.loc[:, "50%_clipped"] = agent_summaries["50%"].clip(
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

    print(agent_summaries["50%"])
    ax.scatter(
        agent_summaries["release_date"],
        agent_summaries["50%_clipped"],
        marker="o",
        edgecolor=[
            (
                src.utils.plots.get_agent_color(
                    plot_params["colors"], agent_summaries["agent"].iloc[i]
                )
                if agent_summaries["50%"].iloc[i] > 1
                else "black"
            )
            for i in range(len(agent_summaries))
        ],
        facecolor=[
            (
                src.utils.plots.get_agent_color(
                    plot_params["colors"], agent_summaries["agent"].iloc[i]
                )
                if agent_summaries["50%"].iloc[i] > 1
                else "white"
            )
            for i in range(len(agent_summaries))
        ],
    )

    # Add arrows for out-of-range points
    mask_out_range = agent_summaries["50%_clipped"] != agent_summaries["50%"]
    ax.scatter(
        agent_summaries.loc[mask_out_range, "release_date"],
        [0.4] * mask_out_range.sum(),  # Place at bottom of visible range
        marker="v",  # Downward pointing triangle
        color="black",
    )

    texts = []
    if weighting == "equal_family_weight":
        # Add agent labels to each point
        # for _idx, row in agent_summaries.iterrows():
        #     texts.append(
        #         ax.text(
        #             row["release_date"] + pd.Timedelta(days=3),  # Offset text to the right
        #             row["50%_clipped"],  # Offset text slightly above point
        #             row["agent"],
        #         )
        #     )
        ax.annotate(
            "Red lines are 2024 data only", (0.2, 0.8), xycoords="axes fraction"
        )

    doubling_times = []
    for method in ["OLS", "WLS"]:
        # fit_color = "blue" if method == "OLS" else "red"
        doubling_times.extend(
            [
                # plot_trendline(ax, agent_summaries, after="2022-01-01", log_scale=True, annotate=False, method=method, fit_color='blue'),
                plot_trendline(
                    ax,
                    agent_summaries,
                    after="2022-06-01",
                    log_scale=True,
                    annotate=False,
                    method=method,
                    fit_color="blue",
                ),
                # plot_trendline(ax, agent_summaries, after="2022-01-01", log_scale=True, annotate=False, clip=False)
                plot_trendline(
                    ax,
                    agent_summaries,
                    after="2023-01-01",
                    log_scale=True,
                    annotate=False,
                    method=method,
                    fit_color="blue",
                ),
                plot_trendline(
                    ax,
                    agent_summaries,
                    after="2024-01-01",
                    log_scale=True,
                    annotate=False,
                    method=method,
                    fit_color="red",
                ),
            ]
        )

    # Adjust text positions to avoid overlaps
    adjust_text(
        texts,
        max_move=7,
    )

    ax.set_yscale("log")
    ax.set_yticks([1 / 60, 1, 2, 4, 8, 15, 30, 60, 120, 60 * 40])
    ax.set_yticklabels(
        [
            "1 sec",
            "1 min",
            "2 min",
            "4 min",
            "8 min",
            "15 min",
            "30 min",
            "1 hr",
            "2 hrs",
            "40 hrs",
        ]
    )
    ax.set_xlabel("Release Date")
    ax.set_xticks(
        [
            min(agent_summaries["release_date"]) + pd.Timedelta(days=365 * n)
            for n in range(0, 6)
        ]
    )
    ax.set_xbound(
        datetime.date(2022, 11, 11),  # type: ignore[reportArgumentType]
        max(agent_summaries["release_date"]) + pd.Timedelta(weeks=52),
    )

    ax.set_ylabel("Time horizon @ 50% chance of success")
    return doubling_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file_prefix", type=str, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--plot-format", type=str, default="png")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    weightings = ["invsqrt_task_weight", "equal_task_weight"]
    fig, axs = plt.subplots(1, 2, width_ratios=[6, 1])
    ax0: matplotlib.axes.Axes = axs[0]
    ax1: matplotlib.axes.Axes = axs[1]
    doubling_times = []

    for weighting in weightings:
        agent_summaries = pd.read_csv(args.input_file_prefix + f"{weighting}.csv")
        print(agent_summaries)
        release_dates = yaml.safe_load(args.release_dates.read_text())
        agent_summaries["release_date"] = agent_summaries["agent"].map(
            release_dates["date"]
        )
        params = dvc.api.params_show(stages="plot_logistic_multiverse")
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

        doubling_times.extend(
            plot_multiverse(
                params["plots"], ax0, agent_summaries, focus_agents, weighting
            )
        )

    ax1.boxplot([doubling_times], vert=True)
    fig.suptitle(
        "Multiverse analysis of time horizon\nParameters: weighting, WLS vs OLS, date range"
    )
    ax1.set_xticklabels(["Doubling times\n(days)"])

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file)
    print(f"Saved plot to {args.output_file}")


if __name__ == "__main__":
    main()
