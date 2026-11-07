"""Tahmin modellerini simülatörün içinde, kapalı döngüde karşılaştırır.

Çalıştırmak için: python -m nwl.compare_forecasters
Model eğitimi deney dönemini görmez. Deney sırasında modeller ölçülen toplam yükü izler.
"""

import numpy as np

from nwl.cell import Cell
from nwl.evaluate_forecasts import MILAN_NORMAL_END_DAY, TRAIN_DAYS, all_models
from nwl.metrics import Metrics, compute_metrics
from nwl.milan import load_cells
from nwl.protocol import SLOT_MINUTES
from nwl.scenarios import MARGIN_DAYS, World, measured_past, run_world
from nwl.traffic import human_demand

SEEDS = range(3)
RUN_DAYS = 10


def worlds(cell: Cell) -> list[World]:
    synthetic = human_demand(cell, TRAIN_DAYS + RUN_DAYS, np.random.default_rng(0))
    result = [World("Sentetik", synthetic, TRAIN_DAYS, RUN_DAYS)]
    for name, load in load_cells(TRAIN_DAYS).items():
        human = np.repeat(load, SLOT_MINUTES) * cell.capacity_mbps
        result.append(World(f"{name}, normal", human, TRAIN_DAYS, RUN_DAYS))
        result.append(World(f"{name}, tatil", human, MILAN_NORMAL_END_DAY, RUN_DAYS))
    return result


def measure(runs: list, cell: Cell) -> list[Metrics]:
    return [compute_metrics(run, cell, MARGIN_DAYS) for run in runs]


def print_row(label: str, metrics: list[Metrics]) -> None:
    peak = np.mean([m.peak_load for m in metrics])
    congested = np.mean([m.congested_minutes_per_day for m in metrics])
    unserved = np.mean([m.human_unserved_percent for m in metrics])
    delay = np.mean([m.agent_delay_minutes for m in metrics])
    violations = np.mean([m.deadline_violation_percent for m in metrics])
    print(
        f"{label:34} {100 * peak:6.0f}% {congested:8.1f} {unserved:8.2f}"
        f" {delay:8.1f} {violations:8.2f}"
    )


def compare_world(world: World, cell: Cell) -> None:
    print(f"\n{world.name}")
    print(f"{'':34} {'Tepe':>7} {'Tık.dk':>8} {'Kayıp%':>8} {'Gecikme':>8} {'İhlal%':>8}")
    for scenario, label in (("a", "(a) Ajan yok"), ("b", "(b) Tahmini yok say")):
        print_row(label, measure([run_world(world, cell, scenario, s) for s in SEEDS], cell))
    for scenario, label in (("c", "(c) Kusursuz tahmin"), ("d", "(d) Kusursuz tahmin")):
        print_row(label, measure([run_world(world, cell, scenario, s) for s in SEEDS], cell))

    past = measured_past(world, cell)
    for model in all_models():
        model.fit(past)
        for scenario in ("c", "d"):
            runs = [
                run_world(world, cell, scenario, s, model=model, past=past) for s in SEEDS
            ]
            print_row(f"({scenario}) {model.name}", measure(runs, cell))


def main() -> None:
    cell = Cell()
    print(f"{len(SEEDS)} tohum, {RUN_DAYS - 2 * MARGIN_DAYS} ölçüm günü")
    for world in worlds(cell):
        compare_world(world, cell)


if __name__ == "__main__":
    main()
