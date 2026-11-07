"""`python -m nwl` komutunun giriş noktası."""

from pathlib import Path

import numpy as np

from nwl.cell import Cell
from nwl.plots import plot_daily_load
from nwl.traffic import STEPS_PER_DAY, human_demand

SEED = 42
DAYS = 14
OUTPUT = Path("results/human_load.png")


def main() -> None:
    cell = Cell()
    demand = human_demand(cell, DAYS, np.random.default_rng(SEED))

    load = cell.utilization(demand)
    congested_minutes = cell.is_congested(demand).sum() / DAYS
    print(f"{DAYS} gün, {cell.capacity_mbps:.0f} Mbps kapasite")
    print(f"Ortalama yük:          %{100 * load.mean():.1f}")
    print(f"Günlük tepe ortalaması: %{100 * load.reshape(-1, STEPS_PER_DAY).max(axis=1).mean():.1f}")
    print(f"Günlük tıkanıklık:     {congested_minutes:.1f} dakika")

    OUTPUT.parent.mkdir(exist_ok=True)
    plot_daily_load(demand, cell, OUTPUT)
    print(f"Grafik: {OUTPUT}")


if __name__ == "__main__":
    main()
