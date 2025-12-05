#!/usr/bin/env python3
"""
Parse simulator logs for different eviction algorithms and cache sizes,
compute hit rates, and generate a comparison plot.

Usage:
    python3 scripts/plot_hit_rates.py \
        --log-dir results \
        --output results/hit_rate_vs_cache_size.png
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


MISS_RATIO_RE = re.compile(r"Average request miss ratio:\s*([0-9.]+)")
LOG_NAME_RE = re.compile(
    r"(?P<algo>[a-zA-Z0-9_]+)_cache_size_(?P<size>[0-9.]+)_hit_rate\.log$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hit rate vs cache size for all algorithms."
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=pathlib.Path("results"),
        help="Directory containing *_cache_size_<size>_hit_rate.log files.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/hit_rate_vs_cache_size.png"),
        help="Path to save the plot image.",
    )
    return parser.parse_args()


def extract_miss_ratio(log_path: pathlib.Path) -> float | None:
    """Return the last miss ratio found in a log file."""
    last_match = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = MISS_RATIO_RE.search(line)
            if match:
                last_match = float(match.group(1))
    return last_match


def collect_data(log_dir: pathlib.Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Collect hit rates keyed by algorithm.

    Returns:
        dict of algo -> list of (cache_size_gb, hit_rate)
    """
    data: Dict[str, List[Tuple[float, float]]] = {}
    for log_path in log_dir.glob("*_cache_size_*_hit_rate.log"):
        match = LOG_NAME_RE.search(log_path.name)
        if not match:
            continue

        algo = match.group("algo")
        cache_size = float(match.group("size"))
        miss_ratio = extract_miss_ratio(log_path)
        if miss_ratio is None:
            continue

        hit_rate = 1.0 - miss_ratio
        data.setdefault(algo, []).append((cache_size, hit_rate))

    # Ensure points are ordered by cache size
    for algo, values in data.items():
        data[algo] = sorted(values, key=lambda x: x[0])
    return data


def plot(data: Dict[str, List[Tuple[float, float]]], output_path: pathlib.Path) -> None:
    if not data:
        raise SystemExit("No valid log files found to plot.")

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    algos = sorted(data.keys())
    
    # Distinct, high-contrast colors that are easy to distinguish
    distinct_colors = [
        "#e6194b",  # red
        "#3cb44b",  # green
        "#4363d8",  # blue
        "#f58231",  # orange
        "#911eb4",  # purple
        "#42d4f4",  # cyan
        "#f032e6",  # magenta
        "#000000",  # black
        "#9a6324",  # brown
        "#808000",  # olive
    ]
    # Different markers for each algorithm
    markers = ["o", "s", "^", "D", "v", "p", "h", "X", "*", "P"]
    
    colors = distinct_colors[:len(algos)]

    for i, algo in enumerate(algos):
        cache_sizes, hit_rates = zip(*data[algo])
        ax.plot(
            cache_sizes,
            hit_rates,
            label=algo.replace("_", " ").upper(),
            color=colors[i],
            linewidth=2.5,
            linestyle="-",
            marker=markers[i % len(markers)],
            markersize=8,
            markerfacecolor=colors[i],
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
        )

    ax.set_title("Hit Rate vs. Cache Size", fontsize=16, color="#111827", pad=15)
    ax.set_xlabel("Cache Size (GB)", fontsize=13, color="#111827")
    ax.set_ylabel("Hit Rate", fontsize=13, color="#111827")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(colors="#111827", labelsize=11)

    # Style grid for a clean modern look
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.9,
        alpha=0.75,
        color="#9ca3af",
    )

    legend = ax.legend(
        frameon=False, fontsize=11, labelcolor="#111827", loc="upper left"
    )
    for text in legend.get_texts():
        text.set_color("#111827")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved plot to: {output_path}")


def main() -> None:
    args = parse_args()
    data = collect_data(args.log_dir)
    for algo, values in sorted(data.items()):
        print(f"{algo}:")
        for cache_size, hit_rate in values:
            print(f"  cache_size={cache_size:>4.1f} GB -> hit_rate={hit_rate:.4f}")
    plot(data, args.output)


if __name__ == "__main__":
    main()
