"""Senaryoları karşılaştırdığımız ölçütler."""

from dataclasses import dataclass

import numpy as np

from nwl.cell import Cell
from nwl.simulation import SimulationResult
from nwl.traffic import STEPS_PER_DAY


@dataclass(frozen=True)
class Metrics:
    peak_load: float  # günlük tepe yüklerin ortalaması
    peak_to_average: float
    congested_minutes_per_day: float
    human_unserved_percent: float  # karşılanamayan insan talebi
    agent_delay_minutes: float  # görevin gelişinden bitişine ortalama süre
    deadline_violation_percent: float


def compute_metrics(
    result: SimulationResult, cell: Cell, margin_days: int = 0
) -> Metrics:
    """Baştaki ve sondaki `margin_days` gün ölçüme girmez."""
    first = margin_days * STEPS_PER_DAY
    last = len(result.offered_mbps) - first
    offered = result.offered_mbps[first:last]
    human = result.human_mbps[first:last]
    load = cell.utilization(offered)
    days = len(load) // STEPS_PER_DAY

    peak_load = load.reshape(days, STEPS_PER_DAY).max(axis=1).mean()
    served_share = cell.capacity_mbps / np.maximum(offered, cell.capacity_mbps)
    unserved = human * (1 - served_share)

    # Son teslim süresi simülasyonun dışına taşan görevler değerlendirilmez
    tasks = result.tasks
    evaluated = (
        (tasks.arrival >= first) & (tasks.arrival < last) & (tasks.deadline <= last + first)
    )
    if evaluated.any():
        # Bitmeyen görev, simülasyonun sonunda bitmiş gibi sayılır
        finish = np.nan_to_num(result.finish[evaluated], nan=last + first)
        delay = (finish - tasks.arrival[evaluated]).mean()
        violated = finish > tasks.deadline[evaluated]
        violation_percent = 100 * violated.mean()
    else:
        delay = violation_percent = 0.0

    return Metrics(
        peak_load=float(peak_load),
        peak_to_average=float(peak_load / load.mean()),
        congested_minutes_per_day=float(cell.is_congested(offered).sum() / days),
        human_unserved_percent=float(100 * unserved.sum() / human.sum()),
        agent_delay_minutes=float(delay),
        deadline_violation_percent=float(violation_percent),
    )
