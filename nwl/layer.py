"""Network Weather Layer ve ajanın görev zamanlayıcısı.

NWL yayını alır, çözer ve zamanlayıcıya sunar. Zamanlayıcı acil görevleri hemen
gönderir, ertelenebilir görevleri son teslim süresinden önceki sakin zamanlara taşır.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from nwl.agents import Tasks
from nwl.base_station import BaseStation, LoadMeter
from nwl.protocol import CONFIDENCE_SCALE, HORIZON_SLOTS, SLOT_MINUTES, ForecastMessage, decode

SAFETY_MARGIN_MINUTES = 30.0
QUIET_MARGIN = 0.05
QUIET_ERROR_WEIGHT = 2.0


@dataclass(frozen=True)
class StartOption:
    """Görevin başlayabileceği bir slot ve o slottaki dakika aralığı."""

    first_minute: int
    last_minute: int
    cautious_load: float  # tahmin artı beklenen hata
    expected_error: float = 0.0


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
                expected_error = (1 - entry.confidence) * CONFIDENCE_SCALE
                cautious_load = entry.predicted_load + expected_error
                options.append(StartOption(first, last, cautious_load, expected_error))
        return options


class BroadcastLoss:
    """Yayınların bazı ajanlara ulaşamaması.

    Her görev için hücrede rastgele bir konum seçilir. Oradaki kayıp olasılığı, görevin
    sahibinin kaç yayını üst üste kaçırdığını belirler.
    """

    def __init__(self, loss_by_position: np.ndarray) -> None:
        self.loss_by_position = loss_by_position

    def missed_broadcasts(self, rng: np.random.Generator) -> int:
        """Üst üste kaçırılan yayın sayısı. Hiç duyulmuyorsa ufuk uzunluğunu döndürür."""
        loss = rng.choice(self.loss_by_position)
        if loss >= 1.0:
            return HORIZON_SLOTS
        return min(int(rng.geometric(1 - loss)) - 1, HORIZON_SLOTS)


Policy = Callable[[list[StartOption], np.random.Generator], int]


def quietest_slot(options: list[StartOption], rng: np.random.Generator) -> int:
    """Senaryo (c): en sakin slotun ilk dakikası. Herkes aynı yeri seçer."""
    return min(options, key=lambda option: option.cautious_load).first_minute


def random_in_quiet_window(
    options: list[StartOption],
    rng: np.random.Generator,
    quiet_margin: float = QUIET_MARGIN,
    error_weight: float = QUIET_ERROR_WEIGHT,
) -> int:
    """Senaryo (d): en sakin yüke yakın tüm slotlar arasından rastgele bir dakika.

    Tahmin belirsizse iki slotun farkı anlamsızdır, bu yüzden pencere beklenen hatayla
    genişler. Yoksa hatalı bir tahminin ürettiği sahte çukura herkes birden yığılır.
    """
    quietest = min(options, key=lambda option: option.cautious_load)
    limit = quietest.cautious_load + quiet_margin + error_weight * quietest.expected_error
    minutes = [
        minute
        for option in options
        if option.cautious_load <= limit
        for minute in range(option.first_minute, option.last_minute + 1)
    ]
    return int(rng.choice(minutes))


def schedule_tasks(
    tasks: Tasks,
    base_station: BaseStation,
    policy: Policy,
    rng: np.random.Generator,
    meter: LoadMeter | None = None,
    loss: BroadcastLoss | None = None,
) -> np.ndarray:
    """Her görevin başlayacağı dakikayı belirler.

    `meter` verilirse başlayan görevler ölçülen yüke eklenir, böylece baz istasyonu
    ajanların kaydırdığı yükü sonraki tahminlerinde görür. `loss` verilirse ajanlar
    yayınları kaçırabilir: elindeki son tahmini kullanır, hiç tahmini yoksa hemen gönderir.
    """
    starts = tasks.arrival.copy()
    durations = tasks.ideal_duration
    if meter is not None:
        for i in np.flatnonzero(tasks.urgent):
            meter.add_task(int(starts[i]), durations[i])
    latest_starts = np.floor(tasks.deadline - tasks.ideal_duration - SAFETY_MARGIN_MINUTES)
    heard: dict[int, NetworkWeatherLayer] = {}

    for i in np.argsort(tasks.arrival, kind="stable"):
        if tasks.urgent[i]:
            continue

        now = int(tasks.arrival[i])
        slot = now // SLOT_MINUTES
        if slot not in heard:
            heard[slot] = NetworkWeatherLayer()
            heard[slot].receive(base_station.broadcast(now))

        missed = loss.missed_broadcasts(rng) if loss is not None else 0
        layer = heard.get(slot - missed)
        options = layer.start_options(now, int(latest_starts[i])) if layer else []
        if options:
            starts[i] = policy(options, rng)
        if meter is not None:
            meter.add_task(int(starts[i]), durations[i])

    return starts
