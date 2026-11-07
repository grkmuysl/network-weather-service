"""Ajan görevleri, simülatör ve metrik testleri."""

import numpy as np
import pytest

from nwl.agents import (
    AGENT_VOLUME_SHARE,
    SECONDS_PER_STEP,
    TASK_RATE_MBPS,
    URGENT_SHARE,
    Tasks,
    generate_tasks,
    no_tasks,
)
from nwl.cell import Cell
from nwl.metrics import compute_metrics
from nwl.plots import plot_scenarios
from nwl.scenarios import run_scenario
from nwl.simulation import simulate
from nwl.traffic import STEPS_PER_DAY, daily_profile


QUIET_DAY = np.full(STEPS_PER_DAY, 10.0)


def one_task(minutes: float, arrival: int = 0, deadline: float = 1000.0) -> Tasks:
    """`minutes` dakikada bitecek büyüklükte tek görev."""
    return Tasks(
        arrival=np.array([arrival]),
        size_mb=np.array([minutes * TASK_RATE_MBPS * SECONDS_PER_STEP]),
        urgent=np.array([False]),
        deadline=np.array([deadline]),
    )


def test_tasks_match_requested_volume_and_urgent_share():
    cell = Cell()
    tasks = generate_tasks(cell, days=30, rng=np.random.default_rng(0))
    human_volume = cell.capacity_mbps * daily_profile().sum() * SECONDS_PER_STEP * 30
    assert tasks.size_mb.sum() / human_volume == pytest.approx(AGENT_VOLUME_SHARE, rel=0.05)
    assert tasks.urgent.mean() == pytest.approx(URGENT_SHARE, abs=0.02)


def test_more_tasks_arrive_in_the_evening_than_at_night():
    tasks = generate_tasks(Cell(), days=30, rng=np.random.default_rng(0))
    hour = tasks.arrival % STEPS_PER_DAY // 60
    assert ((20 <= hour) & (hour < 23)).sum() > 2 * ((2 <= hour) & (hour < 5)).sum()


def test_deferrable_tasks_have_longer_deadlines_than_urgent_tasks():
    tasks = generate_tasks(Cell(), days=5, rng=np.random.default_rng(0))
    slack = tasks.deadline - tasks.arrival - tasks.ideal_duration
    assert slack[~tasks.urgent].min() > slack[tasks.urgent].max()


def test_task_finishes_on_time_in_an_empty_cell():
    result = simulate(Cell(), np.zeros(30), one_task(minutes=5, arrival=2))
    assert result.finish[0] == 7
    assert result.agent_mbps[2:7].tolist() == [TASK_RATE_MBPS] * 5
    assert result.agent_mbps[7:].sum() == 0


def test_task_slows_down_when_the_cell_is_overloaded():
    """Talep kapasitenin iki katıysa görev iki kat uzun sürer."""
    cell = Cell(capacity_mbps=TASK_RATE_MBPS)
    human = np.full(30, TASK_RATE_MBPS)
    result = simulate(cell, human, one_task(minutes=5))
    assert result.finish[0] == 10


def test_task_waits_for_its_scheduled_start():
    result = simulate(Cell(), np.zeros(30), one_task(minutes=5), starts=np.array([20]))
    assert result.finish[0] == 25


def test_unfinished_task_counts_as_a_violation():
    result = simulate(Cell(), QUIET_DAY, one_task(minutes=5000, deadline=1000.0))
    assert np.isnan(result.finish[0])
    assert compute_metrics(result, Cell()).deadline_violation_percent == 100


def test_late_task_counts_as_a_violation_and_delay_is_measured():
    task = one_task(minutes=5, arrival=10, deadline=12.0)
    result = simulate(Cell(), QUIET_DAY, task)
    metrics = compute_metrics(result, Cell())
    assert metrics.deadline_violation_percent == 100
    assert metrics.agent_delay_minutes == 5


def test_tasks_with_deadlines_after_the_end_are_not_evaluated():
    task = one_task(minutes=5, deadline=2 * STEPS_PER_DAY)
    result = simulate(Cell(), QUIET_DAY, task)
    assert compute_metrics(result, Cell()).deadline_violation_percent == 0


def test_load_metrics_on_a_known_day():
    cell = Cell(capacity_mbps=100.0, congestion_threshold=0.8)
    human = np.full(STEPS_PER_DAY, 40.0)
    human[:60] = 90.0
    metrics = compute_metrics(simulate(cell, human, no_tasks()), cell)
    assert metrics.peak_load == pytest.approx(0.9)
    assert metrics.peak_to_average == pytest.approx(0.9 / (human.mean() / 100))
    assert metrics.congested_minutes_per_day == 60
    assert metrics.human_unserved_percent == 0


def test_human_demand_above_capacity_is_unserved():
    cell = Cell(capacity_mbps=100.0)
    metrics = compute_metrics(simulate(cell, np.full(STEPS_PER_DAY, 125.0), no_tasks()), cell)
    assert metrics.human_unserved_percent == pytest.approx(20)


def test_scenarios_share_the_same_human_traffic():
    a = run_scenario("a", Cell(), days=2, seed=3)
    b = run_scenario("b", Cell(), days=2, seed=3)
    assert np.array_equal(a.human_mbps, b.human_mbps)


def test_agents_raise_the_peak_and_congestion():
    cell = Cell()
    a = compute_metrics(run_scenario("a", cell, days=7, seed=0), cell)
    b = compute_metrics(run_scenario("b", cell, days=7, seed=0), cell)
    assert b.peak_load > a.peak_load
    assert b.congested_minutes_per_day > a.congested_minutes_per_day


def test_scenario_plot_is_saved(tmp_path):
    cell = Cell()
    path = tmp_path / "scenarios.png"
    result = run_scenario("b", cell, days=1, seed=0)
    plot_scenarios({"(b)": result.offered_mbps}, cell, path)
    assert path.stat().st_size > 0


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        run_scenario("z", Cell(), days=1, seed=0)
