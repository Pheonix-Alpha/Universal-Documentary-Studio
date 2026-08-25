"""Chart engine.

Renders line/bar/comparison charts from verified numeric data only —
callers must supply the (label, value) series themselves; this engine
never invents numbers.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt  # noqa: E402


def render_line_chart(labels: list[str], values: list[float], title: str, output_path: str,
                       width_px: int = 1920, height_px: int = 1080) -> str:
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax.plot(labels, values, marker="o", linewidth=2.5, color="#e8b45c")
    ax.set_title(title, fontsize=20, color="white")
    _style_dark(fig, ax)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def render_bar_chart(labels: list[str], values: list[float], title: str, output_path: str,
                      width_px: int = 1920, height_px: int = 1080) -> str:
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax.bar(labels, values, color="#5ca0e8")
    ax.set_title(title, fontsize=20, color="white")
    _style_dark(fig, ax)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def render_comparison_chart(labels: list[str], series: dict[str, list[float]], title: str,
                             output_path: str, width_px: int = 1920, height_px: int = 1080) -> str:
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    x = range(len(labels))
    n = len(series)
    bar_w = 0.8 / max(1, n)
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n - 1) / 2) * bar_w
        ax.bar([xi + offset for xi in x], values, width=bar_w, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=20, color="white")
    ax.legend(facecolor="#1e1e2b", labelcolor="white")
    _style_dark(fig, ax)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def _style_dark(fig, ax) -> None:
    fig.patch.set_facecolor("#12141c")
    ax.set_facecolor("#12141c")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
