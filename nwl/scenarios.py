"""Karşılaştırılan senaryolar.

Aynı tohum her senaryoda aynı insan trafiğini ve aynı görevleri üretir, böylece
senaryolar arasındaki fark yalnızca ajanların davranışından gelir.

Baştaki ve sondaki birer gün ölçüme girmez: ilk gecede önceki günden ertelenen
görev yoktur, son gün de tahmin ufku simülasyonun sonuna çarpar.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np

from nwl.agents import Tasks, TaskSettings, generate_tasks, no_tasks
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
from nwl.layer import (
    QUIET_ERROR_WEIGHT,
    QUIET_MARGIN,
    quietest_slot,
    random_in_quiet_window,
    schedule_tasks,
)
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


@dataclass(frozen=True)
class ForecastError:
    """Kusursuz tahmine bilerek eklenen hata."""

    shift_minutes: int = 0  # pozitifse tahmin gerçeğin önünden gider
    noise: float = 0.0
    memory: float = 0.0


def measured_past(
    world: World, cell: Cell, settings: TaskSettings = TaskSettings()
) -> np.ndarray:
    """NWL açılmadan önce ölçülen slot yükü: insanlar artı hemen gönderilen görevler."""
    profile = world.task_profile(cell)
    rng = np.random.default_rng(HISTORY_SEED)
    tasks = generate_tasks(cell, world.start_day, rng, profile, settings)
    meter = LoadMeter(cell.utilization(world.past_mbps), cell)
    for start, duration in zip(tasks.arrival, tasks.ideal_duration):
        meter.add_task(int(start), duration)
    return meter.by_slot()


def _perfect_with_error(
    human_load: np.ndarray, error: ForecastError, rng: np.random.Generator
) -> Forecaster:
    forecaster = PerfectForecaster(np.roll(human_load, -error.shift_minutes))
    if error.noise == 0:
        return forecaster
    slots = len(human_load) // SLOT_MINUTES
    return NoisyForecaster(forecaster, error.noise, error.memory, slots, rng)


def run_world(
    world: World,
    cell: Cell,
    scenario: str,
    seed: int,
    model: LoadModel | None = None,
    past: np.ndarray | None = None,
    error: ForecastError = ForecastError(),
    settings: TaskSettings = TaskSettings(),
    quiet_margin: float = QUIET_MARGIN,
    error_weight: float = QUIET_ERROR_WEIGHT,
) -> SimulationResult:
    """Senaryoyu çalıştırır. Eğitilmiş `model` yoksa kusursuz tahmin kullanılır."""
    task_seed, schedule_seed = np.random.SeedSequence(seed).spawn(2)
    human = world.run_mbps
    if scenario == "a":
        return simulate(cell, human, no_tasks())

    task_rng = np.random.default_rng(task_seed)
    tasks = generate_tasks(cell, world.days, task_rng, world.task_profile(cell), settings)
    if scenario == "b":
        return simulate(cell, human, tasks)

    rng = np.random.default_rng(schedule_seed)
    human_load = cell.utilization(human)
    meter = LoadMeter(human_load, cell)
    if model is not None:
        forecaster = ModelForecaster(model, past, meter)
    else:
        forecaster = _perfect_with_error(human_load, error, rng)

    policy = POLICIES[scenario]
    if scenario == "d":
        policy = partial(policy, quiet_margin=quiet_margin, error_weight=error_weight)
    base_station = BaseStation(forecaster, total_slots=len(human) // SLOT_MINUTES)
    starts = schedule_tasks(tasks, base_station, policy, rng, meter)
    return simulate(cell, human, tasks, starts)
