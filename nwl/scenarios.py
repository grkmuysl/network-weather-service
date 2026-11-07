"""Karşılaştırılan senaryolar.

Aynı tohum her senaryoda aynı insan trafiğini ve aynı görevleri üretir, böylece
senaryolar arasındaki fark yalnızca ajanların davranışından gelir.

Baştaki ve sondaki birer gün ölçüme girmez: ilk gecede önceki günden ertelenen
görev yoktur, son gün de tahmin ufku simülasyonun sonuna çarpar.
"""

import numpy as np

from nwl.agents import generate_tasks, no_tasks
from nwl.base_station import BaseStation, PerfectForecaster
from nwl.cell import Cell
from nwl.layer import quietest_slot, random_in_quiet_window, schedule_tasks
from nwl.protocol import SLOT_MINUTES
from nwl.simulation import SimulationResult, simulate
from nwl.traffic import human_demand

MARGIN_DAYS = 1

SCENARIOS = {
    "a": "(a) Ajan yok",
    "b": "(b) Ajanlar tahmini yok sayıyor",
    "c": "(c) Tahmine uyuyor, rastgelelik yok",
    "d": "(d) Tam NWL",
}

POLICIES = {"c": quietest_slot, "d": random_in_quiet_window}


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
