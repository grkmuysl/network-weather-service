"""Yük tahmin modelleri.

Hepsi aynı şekilde kullanılır: `fit` eğitim serisini görür, `forecast` o ana kadarki
geçmişten önümüzdeki 24 saatin slotlarını tahmin eder. Seriler gece yarısı başlar.
"""

from typing import Protocol

import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA

from nwl.protocol import HORIZON_SLOTS

SLOTS_PER_DAY = HORIZON_SLOTS
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY


class LoadModel(Protocol):
    name: str

    def fit(self, series: np.ndarray) -> None: ...

    def forecast(self, history: np.ndarray) -> np.ndarray: ...


class MovingAverage:
    """Son saatlerin ortalaması. Geleceğin her slotu için aynı değeri verir."""

    name = "Hareketli ortalama"

    def __init__(self, window: int = 8) -> None:
        self.window = window

    def fit(self, series: np.ndarray) -> None:
        pass

    def forecast(self, history: np.ndarray) -> np.ndarray:
        return np.full(HORIZON_SLOTS, history[-self.window :].mean())


class SameTimeYesterday:
    """Dünün aynı saatini tekrar eder. Diğer modellerin geçmesi gereken temel çizgi."""

    name = "Dün aynı saat"

    def fit(self, series: np.ndarray) -> None:
        pass

    def forecast(self, history: np.ndarray) -> np.ndarray:
        return history[-SLOTS_PER_DAY:].copy()


class SameTimeLastWeek:
    """Geçen haftanın aynı gününü tekrar eder. Hafta sonu farkını da yakalar."""

    name = "Geçen hafta aynı saat"

    def fit(self, series: np.ndarray) -> None:
        pass

    def forecast(self, history: np.ndarray) -> np.ndarray:
        week = history[-SLOTS_PER_WEEK:]
        return week[:SLOTS_PER_DAY].copy()


class DeseasonalizedArima:
    """Haftanın her slotunun ortalamasını çıkarır, kalan sapmalara ARIMA uygular.

    Mevsimsel SARIMA uzun döngüde çok yavaş olduğu için bu yol seçildi.
    """

    name = "ARIMA"

    def __init__(self, order: tuple[int, int, int] = (2, 0, 1), memory_days: int = 3) -> None:
        self.order = order
        self.memory = memory_days * SLOTS_PER_DAY

    def _weekly_shape(self, start: int, length: int) -> np.ndarray:
        return self.profile[(start + np.arange(length)) % SLOTS_PER_WEEK]

    def fit(self, series: np.ndarray) -> None:
        slot_of_week = np.arange(len(series)) % SLOTS_PER_WEEK
        totals = np.bincount(slot_of_week, series, minlength=SLOTS_PER_WEEK)
        self.profile = totals / np.bincount(slot_of_week, minlength=SLOTS_PER_WEEK)
        residual = series - self._weekly_shape(0, len(series))
        self.model = ARIMA(residual, order=self.order, trend="n").fit()

    def forecast(self, history: np.ndarray) -> np.ndarray:
        recent = history[-self.memory :]
        start = len(history) - len(recent)
        residual = recent - self._weekly_shape(start, len(recent))
        future = self.model.apply(residual).forecast(HORIZON_SLOTS)
        return future + self._weekly_shape(len(history), HORIZON_SLOTS)


class SmallLstm:
    """Son 24 saatin yükünden ve zamanından sonraki 24 saati tek seferde tahmin eder."""

    name = "LSTM"

    def __init__(self, hidden: int = 32, epochs: int = 40, seed: int = 0) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.seed = seed

    def _inputs(self, window: np.ndarray, start: int) -> torch.Tensor:
        """Her adım için normalize yük, günün saati ve haftanın günü (sinüs, kosinüs)."""
        slots = start + np.arange(len(window))
        day_angle = 2 * np.pi * slots / SLOTS_PER_DAY
        week_angle = 2 * np.pi * slots / SLOTS_PER_WEEK
        scaled = (window - self.mean) / self.std
        features = np.stack(
            [
                scaled,
                np.sin(day_angle),
                np.cos(day_angle),
                np.sin(week_angle),
                np.cos(week_angle),
            ],
            axis=1,
        )
        return torch.tensor(features, dtype=torch.float32)

    def fit(self, series: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        self.mean, self.std = series.mean(), series.std()

        starts = range(0, len(series) - 2 * SLOTS_PER_DAY + 1, 4)
        inputs = torch.stack([self._inputs(series[s : s + SLOTS_PER_DAY], s) for s in starts])
        targets = torch.tensor(
            np.stack([series[s + SLOTS_PER_DAY : s + 2 * SLOTS_PER_DAY] for s in starts]),
            dtype=torch.float32,
        )
        targets = (targets - self.mean) / self.std

        self.lstm = torch.nn.LSTM(5, self.hidden, batch_first=True)
        self.head = torch.nn.Linear(self.hidden, HORIZON_SLOTS)
        parameters = [*self.lstm.parameters(), *self.head.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)

        for _ in range(self.epochs):
            for batch in torch.randperm(len(inputs)).split(32):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(self._predict(inputs[batch]), targets[batch])
                loss.backward()
                optimizer.step()

    def _predict(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return self.head(hidden[-1])

    def forecast(self, history: np.ndarray) -> np.ndarray:
        start = len(history) - SLOTS_PER_DAY
        inputs = self._inputs(history[-SLOTS_PER_DAY:], start).unsqueeze(0)
        with torch.no_grad():
            scaled = self._predict(inputs)[0].numpy()
        return scaled * self.std + self.mean
