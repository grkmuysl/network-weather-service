"""Rapor grafikleri."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from nwl.cell import Cell
from nwl.traffic import STEPS_PER_DAY

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style_day_axes(ax: plt.Axes, title: str, y_label: str, y_max: float) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13)
    ax.set_xlabel("Günün saati", color=TEXT_SECONDARY)
    ax.set_ylabel(y_label, color=TEXT_SECONDARY)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _style_load_axes(ax: plt.Axes, cell: Cell, title: str, y_max: float) -> None:
    threshold = cell.congestion_threshold
    ax.axhline(threshold, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.text(7.0, threshold + 0.015, "Tıkanıklık eşiği", color=TEXT_SECONDARY, fontsize=9)
    _style_day_axes(ax, title, "Yük (kapasitenin oranı)", y_max)


def plot_daily_load(demand_mbps: np.ndarray, cell: Cell, path: Path) -> None:
    """Her günü ince, ortalama günü kalın çizgiyle gösterir."""
    days = cell.utilization(demand_mbps).reshape(-1, STEPS_PER_DAY)
    hours = np.arange(STEPS_PER_DAY) / 60.0

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    for day in days:
        ax.plot(hours, day, color=SERIES[0], alpha=0.18, linewidth=0.7)
    ax.plot(hours, days.mean(axis=0), color=SERIES[0], linewidth=2)
    ax.text(13, 0.5, f"Ortalama gün (ince çizgiler: {len(days)} ayrı gün)",
            color=TEXT_PRIMARY, fontsize=9, ha="center")
    _style_load_axes(ax, cell, "Hücrenin günlük insan yükü", y_max=1.0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(demand_by_scenario: dict[str, np.ndarray], cell: Cell, path: Path) -> None:
    """Her senaryonun ortalama gününü üst üste çizer."""
    hours = np.arange(STEPS_PER_DAY) / 60.0

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    highest = 0.0
    for (label, demand), color in zip(demand_by_scenario.items(), SERIES):
        average_day = cell.utilization(demand).reshape(-1, STEPS_PER_DAY).mean(axis=0)
        ax.plot(hours, average_day, color=color, linewidth=2, label=label)
        highest = max(highest, average_day.max())
    _style_load_axes(ax, cell, "Senaryoların ortalama günlük yükü", y_max=1.1 * highest)
    ax.legend(loc="upper center", frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_daily_shapes(shapes: dict[str, np.ndarray], path: Path) -> None:
    """Ortalama günleri kendi tepelerine bölerek yalnızca şekillerini karşılaştırır."""
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    for (label, shape), color in zip(shapes.items(), SERIES):
        hours = np.arange(len(shape)) * 24.0 / len(shape)
        ax.plot(hours, shape / shape.max(), color=color, linewidth=2, label=label)
    _style_day_axes(ax, "Milano bölgeleri ve sentetik eğrinin günlük şekli",
                    "Yük (günün tepesine oranla)", y_max=1.05)
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _style_small_axes(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=10)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


MAIN_WORLDS = ("Sentetik", "Merkez, normal", "Akşam yoğun, normal", "Akşam yoğun, tatil")
MAIN_ROWS = {
    ("a", "-"): "(a) Ajan yok",
    ("b", "-"): "(b) Tahmini yok say",
    ("c", "ARIMA"): "(c) ARIMA, rastgelelik yok",
    ("d", "Dün aynı saat"): "(d) Dün aynı saat",
    ("d", "ARIMA"): "(d) ARIMA",
    ("d", "LSTM"): "(d) LSTM",
    ("d", "Kusursuz"): "(d) Kusursuz tahmin",
}


def plot_main_results(results: pd.DataFrame, path: Path) -> None:
    """Her trafik serisinde senaryoların ortalama tepe yükünü gösterir."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.2), sharey=True, sharex=True, facecolor=SURFACE)
    means = results.groupby(["world", "scenario", "forecaster"]).peak_load.mean()
    positions = np.arange(len(MAIN_ROWS))[::-1]

    for ax, world in zip(axes, MAIN_WORLDS):
        values = [means[(world, *key)] for key in MAIN_ROWS]
        ax.barh(positions, values, color=SERIES[0], height=0.62)
        for position, value in zip(positions, values):
            ax.text(value + 0.05, position, f"{value:.0%}", va="center", color=TEXT_PRIMARY,
                    fontsize=8, zorder=3, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
        ax.axvline(0.8, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        _style_small_axes(ax, world)
        ax.grid(False)
        ax.set_yticks(positions, list(MAIN_ROWS.values()))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlim(0, 3.9)
    fig.suptitle("Ortalama günlük tepe yük (kesikli çizgi: tıkanıklık eşiği)",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


SWEEP_WORLDS = ("Sentetik", "Merkez, normal", "Akşam yoğun, normal")
SWEEP_GROUPS = (
    "Ajan payı",
    "Acil görev payı",
    "En uzun süre (dk)",
    "Sakin pencere payı",
    "Tıkanıklık eşiği",
)


def plot_sweeps(results: pd.DataFrame, path: Path) -> None:
    """Varsayımlar değiştikçe tıkanıklığın nasıl değiştiğini gösterir."""
    fig, axes = plt.subplots(3, 5, figsize=(12, 7), sharex="col", sharey="row", facecolor=SURFACE)
    labels = {"b": "(b) Tahmini yok say", "d": "(d) Tam NWL, ARIMA"}
    means = results.groupby(["world", "group", "scenario", "value"]).congested_minutes_per_day.mean()

    for row, world in enumerate(SWEEP_WORLDS):
        for column, group in enumerate(SWEEP_GROUPS):
            ax = axes[row, column]
            for (scenario, label), color in zip(labels.items(), SERIES):
                line = means[world, group, scenario]
                ax.plot(line.index, line.values, color=color, linewidth=2, marker="o",
                        markersize=4, label=label)
            _style_small_axes(ax, group if row == 0 else "")
            ax.set_ylim(bottom=0)
        axes[row, 0].set_ylabel(f"{world}\ntıkanık dk/gün", color=TEXT_SECONDARY, fontsize=9)

    axes[0, 0].legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_robustness(results: pd.DataFrame, path: Path) -> None:
    """Tahmini bozunca tepe yükün nasıl değiştiğini (b) senaryosuyla kıyaslar."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True, facecolor=SURFACE)
    errors = results[results.group != "referans"].copy()
    errors["value"] = errors["value"].astype(float)
    means = errors.groupby(["world", "group", "forecaster", "value"]).peak_load.mean()
    reference = results[results.group == "referans"].groupby("world").peak_load.mean()
    lines = (
        ("Bağımsız hata", "Pencere sabit", SERIES[1], "-"),
        ("Uzun süren hata", "Pencere sabit", SERIES[1], ":"),
        ("Bağımsız hata", "Pencere genişliyor", SERIES[0], "-"),
        ("Uzun süren hata", "Pencere genişliyor", SERIES[0], ":"),
    )

    for column, world in enumerate(SWEEP_WORLDS):
        top, bottom = axes[0, column], axes[1, column]
        for group, forecaster, color, style in lines:
            line = means[world, group, forecaster]
            top.plot(line.index, line.values, color=color, linestyle=style, linewidth=2,
                     marker="o", markersize=4, label=f"{group}, {forecaster.lower()}")
        shift = means[world, "Zaman kayması (dk)", "Pencere genişliyor"]
        bottom.plot(shift.index, shift.values, color=SERIES[0], linewidth=2, marker="o", markersize=4)
        for ax in (top, bottom):
            ax.axhline(reference[world], color=TEXT_SECONDARY, linewidth=1, linestyle="--")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        _style_small_axes(top, world)
        _style_small_axes(bottom, "")
        top.set_xlabel("Hatanın standart sapması (yük)", color=TEXT_SECONDARY, fontsize=9)
        bottom.set_xlabel("Tahminin zaman kayması (dk)", color=TEXT_SECONDARY, fontsize=9)

    axes[0, 0].set_ylabel("Tepe yük", color=TEXT_SECONDARY, fontsize=9)
    axes[1, 0].set_ylabel("Tepe yük", color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc="upper center", frameon=False,
                      labelcolor=TEXT_PRIMARY, fontsize=8)
    axes[1, 0].text(0.02, 0.9, "Kesikli çizgi: (b) tahmini yok say",
                    transform=axes[1, 0].transAxes, color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
