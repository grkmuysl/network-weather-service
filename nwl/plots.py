"""Rapor grafikleri."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nwl.cell import Cell
from nwl.traffic import STEPS_PER_DAY

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def plot_daily_load(demand_mbps: np.ndarray, cell: Cell, path: Path) -> None:
    """Her günü ince, ortalama günü kalın çizgiyle gösterir."""
    days = cell.utilization(demand_mbps).reshape(-1, STEPS_PER_DAY)
    hours = np.arange(STEPS_PER_DAY) / 60.0

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for day in days:
        ax.plot(hours, day, color=SERIES, alpha=0.18, linewidth=0.7)
    ax.plot(hours, days.mean(axis=0), color=SERIES, linewidth=2)

    threshold = cell.congestion_threshold
    ax.axhline(threshold, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.text(0.2, threshold + 0.015, "Tıkanıklık eşiği", color=TEXT_SECONDARY, fontsize=9)
    ax.text(13, 0.5, f"Ortalama gün (ince çizgiler: {len(days)} ayrı gün)",
            color=TEXT_PRIMARY, fontsize=9, ha="center")

    ax.set_title("Hücrenin günlük insan yükü", loc="left", color=TEXT_PRIMARY, fontsize=13)
    ax.set_xlabel("Günün saati", color=TEXT_SECONDARY)
    ax.set_ylabel("Yük (kapasitenin oranı)", color=TEXT_SECONDARY)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
