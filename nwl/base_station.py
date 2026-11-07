"""Baz istasyonu: hücrenin yükünü ölçer, tahmin eder ve tahmini yayınlar.

Baz istasyonu insan ve ajan trafiğini ayıramaz, yalnızca toplam yükü görür. Modeller
de bu toplamla çalışır. Ajanlar yük kaydırdıkça ölçüm, sonra da tahmin değişir.
"""

from typing import Protocol

import numpy as np

from nwl.agents import TASK_RATE_MBPS
from nwl.cell import Cell
from nwl.forecast import SLOTS_PER_DAY, LoadModel
from nwl.protocol import (
    CONFIDENCE_SCALE,
    HORIZON_SLOTS,
    SLOT_MINUTES,
    ForecastEntry,
    ForecastMessage,
    encode,
)


ERROR_WINDOW_SLOTS = 2 * SLOTS_PER_DAY
WARMUP_STEP_SLOTS = 4


class LoadMeter:
    """Hücrede ölçülen toplam yük. Görevler başladıkça ajan yükü eklenir.

    Görevlerin tıkanıklıkta uzamasını saymaz; her görev tam hızında gitmiş kabul edilir.
    """

    def __init__(self, human_load_by_minute: np.ndarray, cell: Cell) -> None:
        self.load = human_load_by_minute.copy()
        self.task_load = TASK_RATE_MBPS / cell.capacity_mbps

    def add_task(self, start: int, duration: float) -> None:
        minutes = np.arange(start, min(start + int(np.ceil(duration)), len(self.load)))
        self.load[minutes] += self.task_load * np.minimum(start + duration - minutes, 1.0)

    def by_slot(self) -> np.ndarray:
        return self.load.reshape(-1, SLOT_MINUTES).mean(axis=1)


class Forecaster(Protocol):
    """Her tahmin modelinin uyması gereken arayüz."""

    def predict(self, issued_slot: int, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`issued_slot` anında bilinenlerle `slots` için (yük, güven) döndürür."""
        ...


class PerfectForecaster:
    """Geleceği birebir bilen tahminci: sistemin ulaşabileceği en iyi sonucu verir."""

    def __init__(self, load_by_minute: np.ndarray) -> None:
        self.load_by_slot = load_by_minute.reshape(-1, SLOT_MINUTES).mean(axis=1)

    def predict(self, issued_slot: int, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.load_by_slot[slots], np.ones(len(slots))


class NoisyForecaster:
    """Kusursuz tahmine gürültü ekler. Tahmin hatasına dayanıklılığı ölçmek için."""

    def __init__(
        self, inner: Forecaster, noise: float, rng: np.random.Generator
    ) -> None:
        self.inner = inner
        self.noise = noise
        self.rng = rng

    def predict(self, issued_slot: int, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loads, confidences = self.inner.predict(issued_slot, slots)
        noisy = loads + self.rng.normal(0.0, self.noise, len(slots))
        return np.maximum(noisy, 0.0), confidences


class ModelForecaster:
    """Eğitilmiş modeli ölçülen yükle besler, güveni son hatalardan hesaplar."""

    def __init__(self, model: LoadModel, history: np.ndarray, meter: LoadMeter) -> None:
        self.model = model
        self.meter = meter
        self.start = len(history)
        run_slots = len(meter.load) // SLOT_MINUTES
        self.series = np.concatenate([history, np.zeros(run_slots)])
        self.issued = np.full((len(self.series), HORIZON_SLOTS), np.nan)

        first = self.start - ERROR_WINDOW_SLOTS - HORIZON_SLOTS
        for t in range(first, self.start, WARMUP_STEP_SLOTS):
            self.issued[t] = model.forecast(self.series[:t])

    def _expected_error(self, now: int) -> np.ndarray:
        """Her ufuk için son günlerdeki tahminlerin gerçekleşenden ortalama sapması."""
        origins = np.arange(now - ERROR_WINDOW_SLOTS - HORIZON_SLOTS, now)
        targets = origins[:, None] + np.arange(HORIZON_SLOTS)
        realised = targets < now
        errors = np.abs(self.issued[origins] - self.series[np.minimum(targets, now - 1)])
        return np.nanmean(np.where(realised, errors, np.nan), axis=0)

    def predict(self, issued_slot: int, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        now = self.start + issued_slot
        self.series[self.start : now] = self.meter.by_slot()[:issued_slot]
        self.issued[now] = self.model.forecast(self.series[:now])

        loads = np.maximum(self.issued[now][: len(slots)], 0.0)
        error = self._expected_error(now)[: len(slots)]
        return loads, np.clip(1 - error / CONFIDENCE_SCALE, 0.0, 1.0)


class BaseStation:
    def __init__(self, forecaster: Forecaster, total_slots: int) -> None:
        self.forecaster = forecaster
        self.total_slots = total_slots

    def broadcast(self, minute: int) -> bytes:
        """`minute` anının slotundan başlayarak ufuk boyunca tahmini yayınlar."""
        first_slot = minute // SLOT_MINUTES
        slots = np.arange(first_slot, min(first_slot + HORIZON_SLOTS, self.total_slots))
        loads, confidences = self.forecaster.predict(first_slot, slots)
        entries = tuple(
            ForecastEntry(int(slot), float(load), float(confidence))
            for slot, load, confidence in zip(slots, loads, confidences)
        )
        return encode(ForecastMessage(first_slot * SLOT_MINUTES, entries))
