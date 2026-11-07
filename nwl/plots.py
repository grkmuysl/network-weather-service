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

# Grafikler İngilizce çizilir. Türkçe rapor için `set_language("tr")` çağrılır.
TURKISH = {
    "Hour of day": "Günün saati",
    "Load (share of capacity)": "Yük (kapasitenin oranı)",
    "Congestion threshold": "Tıkanıklık eşiği",
    "Daily human load of the cell": "Hücrenin günlük insan yükü",
    "Average daily load of the scenarios": "Senaryoların ortalama günlük yükü",
    "Daily shape of the Milan cells and the synthetic curve": "Milano hücreleri ve sentetik eğrinin günlük şekli",
    "Load (share of the daily peak)": "Yük (günün tepesine oranla)",
    "Synthetic": "Sentetik",
    "Centre": "Merkez",
    "Average": "Ortalama",
    "Evening": "Akşam yoğun",
    "Evening, holiday": "Akşam yoğun, tatil",
    "(a) No agents": "(a) Ajan yok",
    "(b) Ignore forecast": "(b) Tahmini yok say",
    "(c) ARIMA, no randomisation": "(c) ARIMA, rastgelelik yok",
    "(d) Same time yesterday": "(d) Dün aynı saat",
    "(d) ARIMA": "(d) ARIMA",
    "(d) Small LSTM": "(d) Küçük LSTM",
    "(d) Perfect forecast": "(d) Kusursuz tahmin",
    "(d) Full NWL, ARIMA": "(d) Tam NWL, ARIMA",
    "Mean daily peak load (dashed line: congestion threshold)":
        "Ortalama günlük tepe yük (kesikli çizgi: tıkanıklık eşiği)",
    "Agent share": "Ajan payı",
    "Urgent share": "Acil görev payı",
    "Maximum slack (min)": "En uzun süre (dk)",
    "Fixed quiet margin": "Sakin pencere payı",
    "congested min/day": "tıkanık dk/gün",
    "Congested min/day": "Tıkanık dk/gün",
    "Independent error, fixed window": "Bağımsız hata, sabit pencere",
    "Long-lasting error, fixed window": "Uzun süren hata, sabit pencere",
    "Independent error, widening window": "Bağımsız hata, genişleyen pencere",
    "Long-lasting error, widening window": "Uzun süren hata, genişleyen pencere",
    "Error standard deviation (load)": "Hatanın standart sapması (yük)",
    "Forecast time shift (min)": "Tahminin zaman kayması (dk)",
    "Peak load": "Tepe yük",
    "Dashed line: (b) ignore forecast": "Kesikli çizgi: (b) tahmini yok say",
    "Share of broadcasts lost": "Kaybolan yayın oranı",
}
_language = "en"
_figure_scale = 1.0


def set_language(language: str, figure_scale: float = 1.0) -> None:
    """Dar sayfalar için `figure_scale` küçültülür, böylece yazılar okunur kalır."""
    global _language, _figure_scale
    _language = language
    _figure_scale = figure_scale


def _size(width: float, height: float) -> tuple[float, float]:
    return width * _figure_scale, height * _figure_scale


def _t(text: str) -> str:
    return TURKISH[text] if _language == "tr" else text


def _style_day_axes(ax: plt.Axes, title: str, y_label: str, y_max: float) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13)
    ax.set_xlabel(_t("Hour of day"), color=TEXT_SECONDARY)
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
    ax.text(7.0, threshold + 0.015, _t("Congestion threshold"), color=TEXT_SECONDARY, fontsize=9)
    _style_day_axes(ax, title, _t("Load (share of capacity)"), y_max)


def plot_daily_load(demand_mbps: np.ndarray, cell: Cell, path: Path) -> None:
    """Her günü ince, ortalama günü kalın çizgiyle gösterir."""
    days = cell.utilization(demand_mbps).reshape(-1, STEPS_PER_DAY)
    hours = np.arange(STEPS_PER_DAY) / 60.0

    fig, ax = plt.subplots(figsize=_size(9, 4.5), facecolor=SURFACE)
    for day in days:
        ax.plot(hours, day, color=SERIES[0], alpha=0.18, linewidth=0.7)
    ax.plot(hours, days.mean(axis=0), color=SERIES[0], linewidth=2)
    ax.text(13, 0.5, f"Average day (thin lines: {len(days)} individual days)",
            color=TEXT_PRIMARY, fontsize=9, ha="center")
    _style_load_axes(ax, cell, _t("Daily human load of the cell"), y_max=1.0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(demand_by_scenario: dict[str, np.ndarray], cell: Cell, path: Path) -> None:
    """Her senaryonun ortalama gününü üst üste çizer."""
    hours = np.arange(STEPS_PER_DAY) / 60.0

    fig, ax = plt.subplots(figsize=_size(9, 4.5), facecolor=SURFACE)
    highest = 0.0
    for (label, demand), color in zip(demand_by_scenario.items(), SERIES):
        average_day = cell.utilization(demand).reshape(-1, STEPS_PER_DAY).mean(axis=0)
        ax.plot(hours, average_day, color=color, linewidth=2, label=label)
        highest = max(highest, average_day.max())
    _style_load_axes(ax, cell, _t("Average daily load of the scenarios"), y_max=1.1 * highest)
    ax.legend(loc="upper center", frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_daily_shapes(shapes: dict[str, np.ndarray], path: Path) -> None:
    """Ortalama günleri kendi tepelerine bölerek yalnızca şekillerini karşılaştırır."""
    fig, ax = plt.subplots(figsize=_size(9, 4.5), facecolor=SURFACE)
    for (label, shape), color in zip(shapes.items(), SERIES):
        hours = np.arange(len(shape)) * 24.0 / len(shape)
        ax.plot(hours, shape / shape.max(), color=color, linewidth=2, label=_t(label))
    _style_day_axes(ax, _t("Daily shape of the Milan cells and the synthetic curve"),
                    _t("Load (share of the daily peak)"), y_max=1.05)
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


# Rapor İngilizce olduğu için sonuç grafiklerinin etiketleri de İngilizcedir.
WORLD_LABELS = {
    "Sentetik": "Synthetic",
    "Merkez, normal": "Centre",
    "Akşam yoğun, normal": "Evening",
    "Akşam yoğun, tatil": "Evening, holiday",
}
MAIN_WORLDS = tuple(WORLD_LABELS)
MAIN_ROWS = {
    ("a", "-"): "(a) No agents",
    ("b", "-"): "(b) Ignore forecast",
    ("c", "ARIMA"): "(c) ARIMA, no randomisation",
    ("d", "Dün aynı saat"): "(d) Same time yesterday",
    ("d", "ARIMA"): "(d) ARIMA",
    ("d", "LSTM"): "(d) Small LSTM",
    ("d", "Kusursuz"): "(d) Perfect forecast",
}


def plot_main_results(results: pd.DataFrame, path: Path) -> None:
    """Her trafik serisinde senaryoların ortalama tepe yükünü gösterir."""
    fig, axes = plt.subplots(1, 4, figsize=_size(11, 4.2), sharey=True, sharex=True, facecolor=SURFACE)
    means = results.groupby(["world", "scenario", "forecaster"]).peak_load.mean()
    positions = np.arange(len(MAIN_ROWS))[::-1]

    for ax, world in zip(axes, MAIN_WORLDS):
        values = [means[(world, *key)] for key in MAIN_ROWS]
        ax.barh(positions, values, color=SERIES[0], height=0.62)
        for position, value in zip(positions, values):
            ax.text(value + 0.05, position, f"{value:.0%}", va="center", color=TEXT_PRIMARY,
                    fontsize=8, zorder=3, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
        ax.axvline(0.8, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        _style_small_axes(ax, _t(WORLD_LABELS[world]))
        ax.grid(False)
        ax.set_yticks(positions, [_t(label) for label in MAIN_ROWS.values()])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlim(0, 3.9)
    fig.suptitle(_t("Mean daily peak load (dashed line: congestion threshold)"),
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


SWEEP_WORLDS = ("Sentetik", "Merkez, normal", "Akşam yoğun, normal")
SWEEP_GROUPS = {
    "Ajan payı": "Agent share",
    "Acil görev payı": "Urgent share",
    "En uzun süre (dk)": "Maximum slack (min)",
    "Sakin pencere payı": "Fixed quiet margin",
    "Tıkanıklık eşiği": _t("Congestion threshold"),
}


def plot_sweeps(results: pd.DataFrame, path: Path) -> None:
    """Varsayımlar değiştikçe tıkanıklığın nasıl değiştiğini gösterir."""
    fig, axes = plt.subplots(3, 5, figsize=_size(12, 7), sharex="col", sharey="row", facecolor=SURFACE)
    labels = {"b": "(b) Ignore forecast", "d": "(d) Full NWL, ARIMA"}
    means = results.groupby(["world", "group", "scenario", "value"]).congested_minutes_per_day.mean()

    for row, world in enumerate(SWEEP_WORLDS):
        for column, (group, title) in enumerate(SWEEP_GROUPS.items()):
            ax = axes[row, column]
            for (scenario, label), color in zip(labels.items(), SERIES):
                line = means[world, group, scenario]
                ax.plot(line.index, line.values, color=color, linewidth=2, marker="o",
                        markersize=4, label=_t(label))
            _style_small_axes(ax, _t(title) if row == 0 else "")
            ax.set_ylim(bottom=0)
        axes[row, 0].set_ylabel(f"{_t(WORLD_LABELS[world])}\n{_t('congested min/day')}", color=TEXT_SECONDARY, fontsize=9)

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False,
               labelcolor=TEXT_PRIMARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_robustness(results: pd.DataFrame, path: Path) -> None:
    """Tahmini bozunca tepe yükün nasıl değiştiğini (b) senaryosuyla kıyaslar."""
    fig, axes = plt.subplots(2, 3, figsize=_size(11, 6.5), sharey=True, facecolor=SURFACE)
    errors = results[results.group != "referans"].copy()
    errors["value"] = errors["value"].astype(float)
    means = errors.groupby(["world", "group", "forecaster", "value"]).peak_load.mean()
    reference = results[results.group == "referans"].groupby("world").peak_load.mean()
    lines = (
        ("Bağımsız hata", "Pencere sabit", SERIES[1], "-", "Independent error, fixed window"),
        ("Uzun süren hata", "Pencere sabit", SERIES[1], ":", "Long-lasting error, fixed window"),
        ("Bağımsız hata", "Pencere genişliyor", SERIES[0], "-", "Independent error, widening window"),
        ("Uzun süren hata", "Pencere genişliyor", SERIES[0], ":", "Long-lasting error, widening window"),
    )

    for column, world in enumerate(SWEEP_WORLDS):
        top, bottom = axes[0, column], axes[1, column]
        for group, forecaster, color, style, label in lines:
            line = means[world, group, forecaster]
            top.plot(line.index, line.values, color=color, linestyle=style, linewidth=2,
                     marker="o", markersize=4, label=_t(label))
        shift = means[world, "Zaman kayması (dk)", "Pencere genişliyor"]
        bottom.plot(shift.index, shift.values, color=SERIES[0], linewidth=2, marker="o", markersize=4)
        for ax in (top, bottom):
            ax.axhline(reference[world], color=TEXT_SECONDARY, linewidth=1, linestyle="--")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        _style_small_axes(top, _t(WORLD_LABELS[world]))
        _style_small_axes(bottom, "")
        top.set_xlabel(_t("Error standard deviation (load)"), color=TEXT_SECONDARY, fontsize=9)
        bottom.set_xlabel(_t("Forecast time shift (min)"), color=TEXT_SECONDARY, fontsize=9)

    axes[0, 0].set_ylabel(_t("Peak load"), color=TEXT_SECONDARY, fontsize=9)
    axes[1, 0].set_ylabel(_t("Peak load"), color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc="upper center", frameon=False,
                      labelcolor=TEXT_PRIMARY, fontsize=8)
    axes[1, 0].text(0.02, 0.9, _t("Dashed line: (b) ignore forecast"),
                    transform=axes[1, 0].transAxes, color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_broadcast_loss(results: pd.DataFrame, path: Path) -> None:
    """Yayın kaybı arttıkça tıkanıklığın nasıl değiştiğini gösterir."""
    fig, axes = plt.subplots(1, 3, figsize=_size(11, 3.6), sharex=True, facecolor=SURFACE)
    lost = results[results.group == "Sabit kayıp"].copy()
    lost["value"] = lost["value"].astype(float)
    means = lost.groupby(["world", "value"]).congested_minutes_per_day.mean()
    reference = results[results.group == "referans"].groupby("world")
    reference = reference.congested_minutes_per_day.mean()

    for ax, world in zip(axes, SWEEP_WORLDS):
        line = means[world]
        ax.plot(line.index, line.values, color=SERIES[0], linewidth=2, marker="o", markersize=4)
        ax.axhline(reference[world], color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        _style_small_axes(ax, _t(WORLD_LABELS[world]))
        ax.set_ylim(bottom=0)
        ax.set_xlabel(_t("Share of broadcasts lost"), color=TEXT_SECONDARY, fontsize=9)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[0].set_ylabel(_t("Congested min/day"), color=TEXT_SECONDARY, fontsize=9)
    axes[0].text(0.02, 0.8, _t("Dashed line: (b) ignore forecast"),
                 transform=axes[0].transAxes, color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
