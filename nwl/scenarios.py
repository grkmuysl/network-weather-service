"""Karşılaştırılan senaryolar.

Aynı tohum her senaryoda aynı insan trafiğini ve aynı görevleri üretir, böylece
senaryolar arasındaki fark yalnızca ajanların davranışından gelir.
"""

import numpy as np

from nwl.agents import generate_tasks, no_tasks
from nwl.cell import Cell
from nwl.simulation import SimulationResult, simulate
from nwl.traffic import human_demand

SCENARIOS = {
    "a": "(a) Ajan yok",
    "b": "(b) Ajanlar tahmini yok sayıyor",
}


def run_scenario(scenario: str, cell: Cell, days: int, seed: int) -> SimulationResult:
    human_seed, task_seed = np.random.SeedSequence(seed).spawn(2)
    human = human_demand(cell, days, np.random.default_rng(human_seed))

    if scenario == "a":
        return simulate(cell, human, no_tasks())
    if scenario == "b":
        tasks = generate_tasks(cell, days, np.random.default_rng(task_seed))
        return simulate(cell, human, tasks)
    raise ValueError(f"Bilinmeyen senaryo: {scenario}")
