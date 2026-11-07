"""Baz istasyonu: yükü tahmin eder ve tahmini yayınlar."""

from typing import Protocol

import numpy as np

from nwl.protocol import (
    HORIZON_SLOTS,
    SLOT_MINUTES,
    ForecastEntry,
    ForecastMessage,
    encode,
)


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
