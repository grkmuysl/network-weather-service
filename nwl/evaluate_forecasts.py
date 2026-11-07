"""Tahmin modellerini aynı test günlerinde karşılaştırır.

Çalıştırmak için: python -m nwl.evaluate_forecasts
Model eğitimi test dönemini hiç görmez. Milano verisinde normal günler ile yılbaşı
tatili ayrı ölçülür, çünkü tatil eğitimde hiç görülmemiş bir davranıştır.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nwl.cell import Cell
from nwl.forecast import (
    SLOTS_PER_DAY,
    DeseasonalizedArima,
    LoadModel,
    MovingAverage,
    SameTimeLastWeek,
    SameTimeYesterday,
    SmallLstm,
)
from nwl.milan import load_cells
from nwl.plots import plot_daily_shapes
from nwl.protocol import HORIZON_SLOTS, SLOT_MINUTES
from nwl.traffic import daily_profile, human_demand

SEED = 0
TRAIN_DAYS = 42
TEST_DAYS = 14
ORIGIN_EVERY_SLOTS = 24
HORIZONS = {"15 dk": 1, "1 saat": 4, "6 saat": 24, "24 saat": 96}
MILAN_NORMAL_END_DAY = 52  # 23 Aralık
RESULTS = Path("results")


@dataclass(frozen=True)
class ForecastScore:
    mae: float
    rmse: float
    mae_by_horizon: dict[str, float]


def score_model(model: LoadModel, series: np.ndarray, test_start: int) -> ForecastScore:
    """Test döneminde düzenli aralıklarla 24 saatlik tahmin yapıp hataları ölçer."""
    origins = range(test_start, len(series) - HORIZON_SLOTS + 1, ORIGIN_EVERY_SLOTS)
    errors = np.stack(
        [model.forecast(series[:t]) - series[t : t + HORIZON_SLOTS] for t in origins]
    )
    absolute = np.abs(errors)
    return ForecastScore(
        mae=float(absolute.mean()),
        rmse=float(np.sqrt((errors**2).mean())),
        mae_by_horizon={
            name: float(absolute[:, steps - 1].mean()) for name, steps in HORIZONS.items()
        },
    )


def synthetic_series(days: int, seed: int) -> np.ndarray:
    """Slot başına ortalama insan yükü (kapasitenin oranı olarak)."""
    cell = Cell()
    demand = human_demand(cell, days, np.random.default_rng(seed))
    return cell.utilization(demand).reshape(-1, SLOT_MINUTES).mean(axis=1)


def all_models() -> list[LoadModel]:
    return [
        MovingAverage(),
        SameTimeYesterday(),
        SameTimeLastWeek(),
        DeseasonalizedArima(),
        SmallLstm(),
    ]


def print_header(title: str) -> None:
    print(f"\n{title}")
    print(f"{'Model':24} {'MAE':>7} {'RMSE':>7}  " + " ".join(f"{h:>8}" for h in HORIZONS))


def print_score(name: str, score: ForecastScore) -> None:
    by_horizon = " ".join(f"{score.mae_by_horizon[h]:8.3f}" for h in HORIZONS)
    print(f"{name:24} {score.mae:7.3f} {score.rmse:7.3f}  {by_horizon}")


def evaluate_synthetic() -> None:
    series = synthetic_series(TRAIN_DAYS + TEST_DAYS, SEED)
    test_start = TRAIN_DAYS * SLOTS_PER_DAY
    print_header(f"Sentetik veri: {TRAIN_DAYS} gün eğitim, {TEST_DAYS} gün test")
    for model in all_models():
        model.fit(series[:test_start])
        print_score(model.name, score_model(model, series, test_start))


def compare_shapes(cells: dict[str, np.ndarray]) -> None:
    """Gerçek ortalama günleri sentetik eğriyle kıyaslar."""
    normal_end = MILAN_NORMAL_END_DAY * SLOTS_PER_DAY
    shapes = {
        name: series[:normal_end].reshape(-1, SLOTS_PER_DAY).mean(axis=0)
        for name, series in cells.items()
    }
    synthetic = daily_profile().reshape(-1, SLOT_MINUTES).mean(axis=1)

    print("\nSentetik eğriyle korelasyon")
    for name, shape in shapes.items():
        print(f"{name:24} {np.corrcoef(shape, synthetic)[0, 1]:7.2f}")
    plot_daily_shapes({**shapes, "Sentetik": synthetic}, RESULTS / "milan_shapes.png")


def evaluate_milan() -> None:
    train_end = TRAIN_DAYS * SLOTS_PER_DAY
    normal_end = MILAN_NORMAL_END_DAY * SLOTS_PER_DAY
    cells = load_cells(TRAIN_DAYS)
    compare_shapes(cells)
    for cell, series in cells.items():
        models = all_models()
        for model in models:
            model.fit(series[:train_end])
        print_header(f"Milano, {cell}: normal günler (13-22 Aralık)")
        for model in models:
            print_score(model.name, score_model(model, series[:normal_end], train_end))
        print_header(f"Milano, {cell}: tatil (23 Aralık-1 Ocak)")
        for model in models:
            print_score(model.name, score_model(model, series, normal_end))


def main() -> None:
    evaluate_synthetic()
    evaluate_milan()


if __name__ == "__main__":
    main()
