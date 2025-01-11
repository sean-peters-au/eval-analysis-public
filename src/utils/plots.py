from typing import TypedDict

import matplotlib.axes


class AgentColors(TypedDict):
    base: str
    dark: str
    light: str


class PlotColorsParams(TypedDict):
    agent_aliases: dict[str, AgentColors]
    default: str


class PlotParams(TypedDict):
    colors: PlotColorsParams
    legend_order: list[str]


def format_time_label(seconds: float) -> str:
    hours = seconds / 3600
    if hours >= 24 * 8:
        return f"{int(hours / 24)}d"
    if hours >= 1:
        return f"{int(hours)}h"
    if hours >= 1 / 60:
        return f"{int(hours * 60)}min"
    return f"{int(seconds)}s"


def get_agent_color(colors: PlotColorsParams, agent: str, variant: str = "base") -> str:
    """Get color for agent, falling back to default if not found."""
    if "human" in agent.lower():
        agent = "human"

    return colors["agent_aliases"].get(agent, {}).get(variant, colors["default"])


def create_sorted_legend(ax: matplotlib.axes.Axes, legend_order: list[str]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = sorted(
        zip(handles, labels),
        key=lambda x: legend_order.index(x[1])
        if x[1] in legend_order
        else float("inf"),
    )
    handles, labels = zip(*legend_elements)

    ax.legend(handles=handles, labels=labels)
