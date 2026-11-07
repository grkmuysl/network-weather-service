"""Dakika dakika ilerleyen hücre simülasyonu.

Toplam talep kapasiteyi aşarsa herkes aynı oranda yavaşlar. Ajan görevleri
yetişemeyen veriyi sonraki dakikalarda gönderir. İnsan talebi beklemez: o
dakikada karşılanamayan kısmı karşılanmamış sayılır.
"""

from dataclasses import dataclass

import numpy as np

from nwl.agents import SECONDS_PER_STEP, TASK_RATE_MBPS, Tasks
from nwl.cell import Cell

FINISHED = 1e-9


@dataclass(frozen=True)
class SimulationResult:
    human_mbps: np.ndarray
    agent_mbps: np.ndarray  # ajanların dakika başına istediği hız
    tasks: Tasks
    finish: np.ndarray  # her görevin bittiği dakika, bitmediyse NaN

    @property
    def offered_mbps(self) -> np.ndarray:
        return self.human_mbps + self.agent_mbps


def simulate(
    cell: Cell,
    human_mbps: np.ndarray,
    tasks: Tasks,
    starts: np.ndarray | None = None,
) -> SimulationResult:
    """Görevleri `starts` dakikalarında başlatır. Verilmezse geldikleri an başlarlar."""
    if starts is None:
        starts = tasks.arrival

    order = np.argsort(starts, kind="stable")
    remaining = tasks.size_mb.copy()
    finish = np.full(len(tasks), np.nan)
    agent_mbps = np.zeros(len(human_mbps))
    active: list[int] = []
    next_task = 0

    for t in range(len(human_mbps)):
        while next_task < len(order) and starts[order[next_task]] <= t:
            active.append(order[next_task])
            next_task += 1
        if not active:
            continue

        ids = np.array(active)
        wanted = np.minimum(TASK_RATE_MBPS, remaining[ids] / SECONDS_PER_STEP)
        agent_mbps[t] = wanted.sum()
        served_share = min(1.0, cell.capacity_mbps / (human_mbps[t] + agent_mbps[t]))

        capacity_mb = TASK_RATE_MBPS * served_share * SECONDS_PER_STEP
        remaining[ids] -= np.minimum(remaining[ids], capacity_mb)
        done = remaining[ids] <= FINISHED
        finish[ids[done]] = t + 1
        active = ids[~done].tolist()

    return SimulationResult(human_mbps, agent_mbps, tasks, finish)
