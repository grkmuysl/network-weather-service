"""Karşılaştırılan senaryolar.

Aynı tohum her senaryoda aynı insan trafiğini ve aynı görevleri üretir, böylece
senaryolar arasındaki fark yalnızca ajanların davranışından gelir.

Baştaki ve sondaki birer gün ölçüme girmez: ilk gecede önceki günden ertelenen
görev yoktur, son gün de tahmin ufku simülasyonun sonuna çarpar.
"""

from dataclasses import dataclass

import numpy as np

from nwl.agents import Tasks, generate_tasks, no_tasks
from nwl.base_station import (
    BaseStation,
    Forecaster,
    LoadMeter,
    ModelForecaster,
    NoisyForecaster,
    PerfectForecaster,
)
from nwl.cell import Cell
from nwl.forecast import LoadModel
from nwl.layer import quietest_slot, random_in_quiet_window, schedule_tasks
from nwl.protocol import SLOT_MINUTES
from nwl.simulation import SimulationResult, simulate
from nwl.traffic import STEPS_PER_DAY, human_demand

MARGIN_DAYS = 1

SCENARIOS = {
    "a": "(a) Ajan yok",
    "b": "(b) Ajanlar tahmini yok sayıyor",
    "c": "(c) Tahmine uyuyor, rastgelelik yok",
    "d": "(d) Tam NWL",
}

POLICIES = {"c": quietest_slot, "d": random_in_quiet_window}
HISTORY_SEED = 12345


def run_scenario(scenario: str, cell: Cell, days: int, seed: int) -> SimulationResult:
    """`days` ölçüm günü ve iki kenar günü için senaryoyu çalıştırır."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Bilinmeyen senaryo: {scenario}")

    total_days = days + 2 * MARGIN_DAYS
    human_seed, task_seed, schedule_seed = np.random.SeedSequence(seed).spawn(3)
    human = human_demand(cell, total_days, np.random.default_rng(human_seed))

    if scenario == "a":
        return simulate(cell, human, no_tasks())

    tasks = generate_tasks(cell, total_days, np.random.default_rng(task_seed))
    if scenario == "b":
        return simulate(cell, human, tasks)

    forecaster = PerfectForecaster(cell.utilization(human))
    base_station = BaseStation(forecaster, total_slots=len(human) // SLOT_MINUTES)
    rng = np.random.default_rng(schedule_seed)
    starts = schedule_tasks(tasks, base_station, POLICIES[scenario], rng)
    return simulate(cell, human, tasks, starts)


@dataclass(frozen=True)
class World:
    """Bir insan trafiği serisi. `start_day` öncesi geçmiş, sonrası deney dönemidir."""

    name: str
    human_mbps: np.ndarray
    start_day: int
    days: int

    @property
    def past_mbps(self) -> np.ndarray:
        return self.human_mbps[: self.start_day * STEPS_PER_DAY]

    @property
    def run_mbps(self) -> np.ndarray:
        first = self.start_day * STEPS_PER_DAY
        return self.human_mbps[first : first + self.days * STEPS_PER_DAY]

    def task_profile(self, cell: Cell) -> np.ndarray:
        """Görevler bu hücrenin geçmişteki ortalama gününe göre gelir."""
        return cell.utilization(self.past_mbps).reshape(-1, STEPS_PER_DAY).mean(axis=0)


def measured_past(world: World, cell: Cell) -> np.ndarray:
    """NWL açılmadan önce ölçülen slot yükü: insanlar artı hemen gönderilen görevler."""
    profile = world.task_profile(cell)
    rng = np.random.default_rng(HISTORY_SEED)
    tasks = generate_tasks(cell, world.start_day, rng, profile)
    meter = LoadMeter(cell.utilization(world.past_mbps), cell)
    for start, duration in zip(tasks.arrival, tasks.ideal_duration):
        meter.add_task(int(start), duration)
    return meter.by_slot()


def _scheduled_run(
    world: World,
    cell: Cell,
    scenario: str,
    tasks: Tasks,
    forecaster: Forecaster,
    meter: LoadMeter,
    rng: np.random.Generator,
) -> SimulationResult:
    human = world.run_mbps
    base_station = BaseStation(forecaster, total_slots=len(human) // SLOT_MINUTES)
    starts = schedule_tasks(tasks, base_station, POLICIES[scenario], rng, meter)
    return simulate(cell, human, tasks, starts)


def run_world(
    world: World,
    cell: Cell,
    scenario: str,
    seed: int,
    model: LoadModel | None = None,
    past: np.ndarray | None = None,
    noise: float = 0.0,
) -> SimulationResult:
    """Senaryoyu çalıştırır. Eğitilmiş `model` yoksa kusursuz tahmin kullanılır."""
    task_seed, schedule_seed = np.random.SeedSequence(seed).spawn(2)
    human = world.run_mbps
    if scenario == "a":
        return simulate(cell, human, no_tasks())

    profile = world.task_profile(cell)
    tasks = generate_tasks(cell, world.days, np.random.default_rng(task_seed), profile)
    if scenario == "b":
        return simulate(cell, human, tasks)

    rng = np.random.default_rng(schedule_seed)
    human_load = cell.utilization(human)
    meter = LoadMeter(human_load, cell)
    if model is not None:
        forecaster = ModelForecaster(model, past, meter)
    else:
        forecaster = PerfectForecaster(human_load)
        if noise > 0:
            forecaster = NoisyForecaster(forecaster, noise, rng)
    return _scheduled_run(world, cell, scenario, tasks, forecaster, meter, rng)
