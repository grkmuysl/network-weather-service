"""`python -m nwl` komutunun giriş noktası."""

from pathlib import Path

import numpy as np

from nwl.cell import Cell
from nwl.metrics import Metrics, compute_metrics
from nwl.plots import plot_daily_load, plot_scenarios
from nwl.scenarios import MARGIN_DAYS, SCENARIOS, run_scenario
from nwl.traffic import STEPS_PER_DAY

SEEDS = range(10)
DAYS = 14
RESULTS = Path("results")

COLUMNS = {
    "peak_load": "Tepe yük",
    "peak_to_average": "Tepe/ortalama",
    "congested_minutes_per_day": "Tıkanıklık (dk/gün)",
    "human_unserved_percent": "Karşılanmayan insan talebi (%)",
    "agent_delay_minutes": "Ajan gecikmesi (dk)",
    "deadline_violation_percent": "Son teslim ihlali (%)",
}


def print_table(metrics_by_scenario: dict[str, list[Metrics]]) -> None:
    """Her ölçüt için tohumların ortalamasını ve standart sapmasını yazdırır."""
    for label, runs in metrics_by_scenario.items():
        print(f"\n{label}")
        for field, name in COLUMNS.items():
            values = [getattr(run, field) for run in runs]
            print(f"  {name:32} {np.mean(values):8.2f} ± {np.std(values):.2f}")


def main() -> None:
    cell = Cell()
    metrics_by_scenario = {}
    demand_by_scenario = {}

    for scenario, label in SCENARIOS.items():
        results = [run_scenario(scenario, cell, DAYS, seed) for seed in SEEDS]
        metrics_by_scenario[label] = [
            compute_metrics(result, cell, MARGIN_DAYS) for result in results
        ]
        margin = MARGIN_DAYS * STEPS_PER_DAY
        demand_by_scenario[label] = np.concatenate(
            [result.offered_mbps[margin:-margin] for result in results]
        )

    print(f"{len(SEEDS)} tohum, tohum başına {DAYS} gün")
    print_table(metrics_by_scenario)

    RESULTS.mkdir(exist_ok=True)
    plot_daily_load(run_scenario("a", cell, DAYS, SEEDS[0]).human_mbps, cell, RESULTS / "human_load.png")
    plot_scenarios(demand_by_scenario, cell, RESULTS / "scenarios.png")
    print(f"\nGrafikler: {RESULTS}/")


if __name__ == "__main__":
    main()
