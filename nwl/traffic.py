"""Sentetik insan trafiği üretimi.

Simülasyon 1 dakikalık adımlarla ilerler. Günlük şekil iki tepeden oluşur:
gün ortasında geniş bir tepe ve 20:00-23:00 çevresinde dar bir akşam tepesi.
"""

import numpy as np

from nwl.cell import Cell

STEPS_PER_DAY = 1440

NIGHT_LEVEL = 0.15
MIDDAY_HOUR, MIDDAY_HEIGHT, MIDDAY_WIDTH = 13.0, 0.30, 3.5
EVENING_HOUR, EVENING_HEIGHT, EVENING_WIDTH = 21.5, 0.58, 1.8

DAY_VARIATION = 0.05
NOISE_LEVEL = 0.05
NOISE_MEMORY = 0.98


def _bump(hours: np.ndarray, center: float, width: float) -> np.ndarray:
    """Merkezi `center` saati olan çan eğrisi. Gece yarısında kesintisizdir."""
    distance = np.abs(hours - center)
    distance = np.minimum(distance, 24.0 - distance)
    return np.exp(-0.5 * (distance / width) ** 2)


def daily_profile() -> np.ndarray:
    """Gürültüsüz ortalama bir günün yükü (kapasitenin oranı olarak)."""
    hours = np.arange(STEPS_PER_DAY) / 60.0
    return (
        NIGHT_LEVEL
        + MIDDAY_HEIGHT * _bump(hours, MIDDAY_HOUR, MIDDAY_WIDTH)
        + EVENING_HEIGHT * _bump(hours, EVENING_HOUR, EVENING_WIDTH)
    )


def _correlated_noise(steps: int, rng: np.random.Generator) -> np.ndarray:
    """Her adımın bir öncekine benzediği AR(1) gürültüsü."""
    shocks = rng.normal(0.0, NOISE_LEVEL * np.sqrt(1 - NOISE_MEMORY**2), steps)
    noise = np.empty(steps)
    noise[0] = rng.normal(0.0, NOISE_LEVEL)
    for t in range(1, steps):
        noise[t] = NOISE_MEMORY * noise[t - 1] + shocks[t]
    return noise


def human_demand(cell: Cell, days: int, rng: np.random.Generator) -> np.ndarray:
    """`days` gün için dakika başına insan talebi (Mbps)."""
    profile = np.tile(daily_profile(), days)
    day_factors = np.repeat(rng.normal(1.0, DAY_VARIATION, days), STEPS_PER_DAY)
    noise = _correlated_noise(days * STEPS_PER_DAY, rng)
    return np.maximum(cell.capacity_mbps * profile * day_factors * (1 + noise), 0.0)
