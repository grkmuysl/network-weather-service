"""Yapay zekâ ajanlarının görevleri.

Ajanlar kullanıcılar adına çalıştığı için görevler insan trafiğiyle aynı günlük
şekilde gelir: akşam çok, gece az. Bu yüzden ajan trafiği akşam tepesine eklenir.
"""

from dataclasses import dataclass

import numpy as np

from nwl.cell import Cell
from nwl.traffic import STEPS_PER_DAY, daily_profile

SECONDS_PER_STEP = 60

AGENT_VOLUME_SHARE = 0.25
URGENT_SHARE = 0.3
TASK_RATE_MBPS = 2.0
MEAN_DURATION_MINUTES = 10.0
DURATION_SPREAD = 0.8
URGENT_SLACK_MINUTES = 10.0
DEFERRABLE_SLACK_MINUTES = (120.0, 720.0)


@dataclass(frozen=True)
class TaskSettings:
    """Deneylerde değiştirilen görev varsayımları."""

    volume_share: float = AGENT_VOLUME_SHARE
    urgent_share: float = URGENT_SHARE
    deferrable_slack_minutes: tuple[float, float] = DEFERRABLE_SLACK_MINUTES


@dataclass(frozen=True)
class Tasks:
    """Görev listesi. Her dizinin i. elemanı i. göreve aittir."""

    arrival: np.ndarray  # görevin ortaya çıktığı dakika
    size_mb: np.ndarray  # gönderilecek veri (megabit)
    urgent: np.ndarray  # True ise beklemeden gönderilir
    deadline: np.ndarray  # görevin bitmesi gereken son dakika

    def __len__(self) -> int:
        return len(self.arrival)

    @property
    def ideal_duration(self) -> np.ndarray:
        """Hücre hiç tıkanmasaydı görevin süreceği dakika."""
        return self.size_mb / (TASK_RATE_MBPS * SECONDS_PER_STEP)


def no_tasks() -> Tasks:
    return Tasks(
        arrival=np.empty(0, dtype=int),
        size_mb=np.empty(0),
        urgent=np.empty(0, dtype=bool),
        deadline=np.empty(0),
    )


def generate_tasks(
    cell: Cell,
    days: int,
    rng: np.random.Generator,
    profile: np.ndarray | None = None,
    settings: TaskSettings = TaskSettings(),
) -> Tasks:
    """`days` gün için görevleri üretir. Hacim, insan trafiğinin belirli bir oranıdır.

    `profile`, dakika başına ortalama insan yüküdür; verilmezse sentetik eğri kullanılır.
    """
    if profile is None:
        profile = daily_profile()
    mean_size_mb = TASK_RATE_MBPS * SECONDS_PER_STEP * MEAN_DURATION_MINUTES
    human_volume_mb = cell.capacity_mbps * profile * SECONDS_PER_STEP
    arrival_rate = settings.volume_share * human_volume_mb / mean_size_mb

    counts = rng.poisson(np.tile(arrival_rate, days))
    arrival = np.repeat(np.arange(days * STEPS_PER_DAY), counts)
    n = len(arrival)

    # Ortalaması MEAN_DURATION_MINUTES olan, çoğu kısa birkaçı uzun süreler
    log_mean = np.log(MEAN_DURATION_MINUTES) - DURATION_SPREAD**2 / 2
    duration = rng.lognormal(log_mean, DURATION_SPREAD, n)
    size_mb = duration * TASK_RATE_MBPS * SECONDS_PER_STEP

    urgent = rng.random(n) < settings.urgent_share
    slack = np.where(
        urgent,
        URGENT_SLACK_MINUTES,
        rng.uniform(*settings.deferrable_slack_minutes, n),
    )
    return Tasks(arrival, size_mb, urgent, arrival + duration + slack)
