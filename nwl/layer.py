"""Network Weather Layer ve ajanın görev zamanlayıcısı.

NWL yayını alır, çözer ve zamanlayıcıya sunar. Zamanlayıcı acil görevleri hemen
gönderir, ertelenebilir görevleri son teslim süresinden önceki sakin zamanlara taşır.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from nwl.agents import Tasks
from nwl.base_station import BaseStation
from nwl.protocol import SLOT_MINUTES, ForecastMessage, decode

SAFETY_MARGIN_MINUTES = 30.0
QUIET_MARGIN = 0.05


@dataclass(frozen=True)
class StartOption:
    """Görevin başlayabileceği bir slot ve o slottaki dakika aralığı."""

    first_minute: int
    last_minute: int
    predicted_load: float


class NetworkWeatherLayer:
    def __init__(self) -> None:
        self.forecast: ForecastMessage | None = None

    def receive(self, payload: bytes) -> None:
        self.forecast = decode(payload)

    def start_options(self, now: int, latest_start: int) -> list[StartOption]:
        """`now` ile `latest_start` arasında başlanabilecek slotları döndürür."""
        options = []
        for entry in self.forecast.entries:
            slot_start = entry.time_slot * SLOT_MINUTES
            first = max(slot_start, now)
            last = min(slot_start + SLOT_MINUTES - 1, latest_start)
            if first <= last:
                options.append(StartOption(first, last, entry.predicted_load))
        return options


Policy = Callable[[list[StartOption], np.random.Generator], int]


def quietest_slot(options: list[StartOption], rng: np.random.Generator) -> int:
    """Senaryo (c): en sakin slotun ilk dakikası. Herkes aynı yeri seçer."""
    return min(options, key=lambda option: option.predicted_load).first_minute


def random_in_quiet_window(options: list[StartOption], rng: np.random.Generator) -> int:
    """Senaryo (d): en sakin yüke yakın tüm slotlar arasından rastgele bir dakika."""
    lowest = min(option.predicted_load for option in options)
    minutes = [
        minute
        for option in options
        if option.predicted_load <= lowest + QUIET_MARGIN
        for minute in range(option.first_minute, option.last_minute + 1)
    ]
    return int(rng.choice(minutes))


def schedule_tasks(
    tasks: Tasks,
    base_station: BaseStation,
    policy: Policy,
    rng: np.random.Generator,
) -> np.ndarray:
    """Her görevin başlayacağı dakikayı belirler."""
    starts = tasks.arrival.copy()
    latest_starts = np.floor(tasks.deadline - tasks.ideal_duration - SAFETY_MARGIN_MINUTES)
    layer = NetworkWeatherLayer()
    heard_slot = None

    for i in np.argsort(tasks.arrival, kind="stable"):
        if tasks.urgent[i]:
            continue

        now = int(tasks.arrival[i])
        if now // SLOT_MINUTES != heard_slot:
            layer.receive(base_station.broadcast(now))
            heard_slot = now // SLOT_MINUTES

        options = layer.start_options(now, int(latest_starts[i]))
        if options:
            starts[i] = policy(options, rng)

    return starts
