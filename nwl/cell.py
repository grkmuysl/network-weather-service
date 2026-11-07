"""Tek bir baz istasyonu hücresinin modeli."""

from dataclasses import dataclass

import numpy as np


CALIBRATED_CAPACITY_MBPS = 132.0  # Sionna, 10 etkin kullanıcı (calibration/cell.json)


@dataclass(frozen=True)
class Cell:
    """Sabit kapasiteli hücre. Yük, talebin kapasiteye oranıdır."""

    capacity_mbps: float = CALIBRATED_CAPACITY_MBPS
    congestion_threshold: float = 0.8

    def utilization(self, demand_mbps: np.ndarray) -> np.ndarray:
        return np.asarray(demand_mbps) / self.capacity_mbps

    def is_congested(self, demand_mbps: np.ndarray) -> np.ndarray:
        return self.utilization(demand_mbps) > self.congestion_threshold
