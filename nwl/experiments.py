"""Raporun deneyleri. Her deney sonuçlarını results/ altına CSV olarak yazar.

Çalıştırmak için: python -m nwl.experiments main
Diğer deneyler: sweeps, errors, broadcast. Grafikler için: python -m nwl.experiments plots
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from nwl.agents import TaskSettings
from nwl.cell import Cell
from nwl.compare_forecasters import worlds
from nwl.evaluate_forecasts import all_models
from nwl.forecast import DeseasonalizedArima
from nwl.layer import BroadcastLoss
from nwl.metrics import compute_metrics
from nwl.plots import (
    plot_broadcast_loss,
    plot_error_robustness,
    plot_main_results,
    plot_sweeps,
)
from nwl.scenarios import MARGIN_DAYS, ForecastError, measured_past, run_world

RESULTS = Path("results")
MAIN_SEEDS = range(10)
SWEEP_SEEDS = range(5)
SWEEP_WORLDS = ("Sentetik", "Merkez, normal", "Akşam yoğun, normal")

AGENT_SHARES = (0.1, 0.25, 0.5, 0.75)
URGENT_SHARES = (0.1, 0.3, 0.5, 0.7)
SLACK_RANGES = ((60.0, 240.0), (120.0, 720.0), (240.0, 1440.0))
QUIET_MARGINS = (0.0, 0.02, 0.05, 0.1, 0.2)
CONGESTION_THRESHOLDS = (0.7, 0.8, 0.9)

SHIFTS_MINUTES = (-120, -60, -30, 0, 30, 60, 120)
NOISE_LEVELS = (0.05, 0.1, 0.2, 0.3)
NOISE_MEMORIES = (0.0, 0.95)

BROADCAST_PATH = Path("calibration/broadcast.json")
CELL_PATH = Path("calibration/cell.json")
UNIFORM_LOSSES = (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)


def row(world: str, group: str, value: object, scenario: str, forecaster: str, seed: int,
        result, cell: Cell) -> dict:
    return {
        "world": world,
        "group": group,
        "value": value,
        "scenario": scenario,
        "forecaster": forecaster,
        "seed": seed,
        **asdict(compute_metrics(result, cell, MARGIN_DAYS)),
    }


def run_main() -> pd.DataFrame:
    """Tüm senaryolar, tüm modeller, tüm trafik serileri."""
    cell = Cell()
    rows = []
    for world in worlds(cell):
        for scenario in "abcd":
            forecaster = "Kusursuz" if scenario in "cd" else "-"
            for seed in MAIN_SEEDS:
                result = run_world(world, cell, scenario, seed)
                rows.append(row(world.name, "ana", "", scenario, forecaster, seed, result, cell))

        past = measured_past(world, cell)
        for model in all_models():
            model.fit(past)
            for scenario in "cd":
                for seed in MAIN_SEEDS:
                    result = run_world(world, cell, scenario, seed, model=model, past=past)
                    rows.append(
                        row(world.name, "ana", "", scenario, model.name, seed, result, cell)
                    )
            for seed in MAIN_SEEDS:
                result = run_world(
                    world, cell, "d", seed, model=model, past=past, error_weight=0.0
                )
                label = f"{model.name} (pencere sabit)"
                rows.append(row(world.name, "ana", "", "d", label, seed, result, cell))
    return pd.DataFrame(rows)


def run_sweeps() -> pd.DataFrame:
    """Varsayımları tek tek değiştirir. Tahminci ARIMA, karşılaştırma senaryo (b)."""
    cell = Cell()
    settings_by_group = {
        "Ajan payı": {s: TaskSettings(volume_share=s) for s in AGENT_SHARES},
        "Acil görev payı": {s: TaskSettings(urgent_share=s) for s in URGENT_SHARES},
        "En uzun süre (dk)": {
            s[1]: TaskSettings(deferrable_slack_minutes=s) for s in SLACK_RANGES
        },
    }
    rows = []
    for world in worlds(cell):
        if world.name not in SWEEP_WORLDS:
            continue

        for group, levels in settings_by_group.items():
            for value, settings in levels.items():
                past = measured_past(world, cell, settings)
                model = DeseasonalizedArima()
                model.fit(past)
                for seed in SWEEP_SEEDS:
                    baseline = run_world(world, cell, "b", seed, settings=settings)
                    rows.append(row(world.name, group, value, "b", "-", seed, baseline, cell))
                    result = run_world(
                        world, cell, "d", seed, model=model, past=past, settings=settings
                    )
                    rows.append(row(world.name, group, value, "d", "ARIMA", seed, result, cell))

        past = measured_past(world, cell)
        model = DeseasonalizedArima()
        model.fit(past)
        for seed in SWEEP_SEEDS:
            baseline = run_world(world, cell, "b", seed)
            result = run_world(world, cell, "d", seed, model=model, past=past)
            for threshold in CONGESTION_THRESHOLDS:
                measured = Cell(congestion_threshold=threshold)
                group = "Tıkanıklık eşiği"
                rows.append(row(world.name, group, threshold, "b", "-", seed, baseline, measured))
                rows.append(row(world.name, group, threshold, "d", "ARIMA", seed, result, measured))
            for margin in QUIET_MARGINS:
                changed = run_world(
                    world, cell, "d", seed, model=model, past=past, quiet_margin=margin
                )
                rows.append(row(world.name, "Sakin pencere payı", margin, "b", "-", seed,
                                baseline, cell))
                rows.append(row(world.name, "Sakin pencere payı", margin, "d", "ARIMA", seed,
                                changed, cell))
    return pd.DataFrame(rows)


def run_errors() -> pd.DataFrame:
    """Kusursuz tahmini bilerek bozar: zaman kayması ve kalıcı hata eğrileri."""
    cell = Cell()
    errors = {("Zaman kayması (dk)", s): ForecastError(shift_minutes=s) for s in SHIFTS_MINUTES}
    for memory in NOISE_MEMORIES:
        name = "Bağımsız hata" if memory == 0 else "Uzun süren hata"
        for noise in NOISE_LEVELS:
            errors[(name, noise)] = ForecastError(noise=noise, memory=memory)

    rows = []
    for world in worlds(cell):
        if world.name not in SWEEP_WORLDS:
            continue
        for seed in MAIN_SEEDS:
            baseline = run_world(world, cell, "b", seed)
            rows.append(row(world.name, "referans", "", "b", "-", seed, baseline, cell))
            for (group, value), error in errors.items():
                result = run_world(world, cell, "d", seed, error=error)
                rows.append(row(world.name, group, value, "d", "Pencere genişliyor", seed,
                                result, cell))
                fixed = run_world(world, cell, "d", seed, error=error, error_weight=0.0)
                rows.append(row(world.name, group, value, "d", "Pencere sabit", seed, fixed, cell))
    return pd.DataFrame(rows)


def loss_by_position(curve: list[float], snr_db: list[float]) -> np.ndarray:
    """Sionna'nın kayıp eğrisini hücredeki SINR dağılımına uygular."""
    sinr_db = json.loads(CELL_PATH.read_text())["sinr_db_percentiles"]
    return np.interp(sinr_db, snr_db, curve)


def run_broadcast() -> pd.DataFrame:
    """Yayınlar kaybolunca sistemin nasıl davrandığını ölçer. Tahminci ARIMA."""
    cell = Cell()
    calibration = json.loads(BROADCAST_PATH.read_text())
    losses = {("Sabit kayıp", p): BroadcastLoss(np.array([p])) for p in UNIFORM_LOSSES}
    for copies, curve in calibration["message_error_rate"].items():
        positions = loss_by_position(curve, calibration["snr_db"])
        losses[("Sionna", int(copies))] = BroadcastLoss(positions)

    rows = []
    for world in worlds(cell):
        if world.name not in SWEEP_WORLDS:
            continue
        past = measured_past(world, cell)
        model = DeseasonalizedArima()
        model.fit(past)
        for seed in MAIN_SEEDS:
            baseline = run_world(world, cell, "b", seed)
            rows.append(row(world.name, "referans", "", "b", "-", seed, baseline, cell))
            for (group, value), loss in losses.items():
                result = run_world(world, cell, "d", seed, model=model, past=past, loss=loss)
                rows.append(row(world.name, group, value, "d", "ARIMA", seed, result, cell))
    return pd.DataFrame(rows)


def make_plots() -> None:
    plot_main_results(pd.read_csv(RESULTS / "main.csv"), RESULTS / "main_results.png")
    plot_sweeps(pd.read_csv(RESULTS / "sweeps.csv"), RESULTS / "sweeps.png")
    plot_error_robustness(pd.read_csv(RESULTS / "errors.csv"), RESULTS / "robustness.png")
    plot_broadcast_loss(pd.read_csv(RESULTS / "broadcast.csv"), RESULTS / "broadcast_loss.png")


EXPERIMENTS = {
    "main": run_main,
    "sweeps": run_sweeps,
    "errors": run_errors,
    "broadcast": run_broadcast,
}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    name = sys.argv[1]
    if name == "plots":
        make_plots()
    else:
        EXPERIMENTS[name]().to_csv(RESULTS / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
