"""Hücre modeli ve insan trafiği testleri."""

import numpy as np

from nwl.cell import Cell
from nwl.plots import plot_daily_load
from nwl.traffic import STEPS_PER_DAY, daily_profile, human_demand


def test_utilization_is_demand_over_capacity():
    cell = Cell(capacity_mbps=200.0)
    assert cell.utilization(np.array([50.0]))[0] == 0.25


def test_congestion_starts_above_threshold():
    cell = Cell(capacity_mbps=100.0, congestion_threshold=0.8)
    assert cell.is_congested(np.array([80.0, 81.0])).tolist() == [False, True]


def test_profile_peaks_in_the_evening_and_bottoms_out_at_night():
    profile = daily_profile()
    assert 20 * 60 <= profile.argmax() < 23 * 60
    assert 2 * 60 <= profile.argmin() < 7 * 60


def test_profile_has_no_jump_at_midnight():
    profile = daily_profile()
    assert abs(profile[0] - profile[-1]) < 0.01


def test_demand_has_one_value_per_minute():
    demand = human_demand(Cell(), days=3, rng=np.random.default_rng(0))
    assert demand.shape == (3 * STEPS_PER_DAY,)
    assert (demand >= 0).all()


def test_same_seed_gives_same_traffic():
    first = human_demand(Cell(), days=2, rng=np.random.default_rng(7))
    second = human_demand(Cell(), days=2, rng=np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_different_seeds_give_different_traffic():
    first = human_demand(Cell(), days=2, rng=np.random.default_rng(1))
    second = human_demand(Cell(), days=2, rng=np.random.default_rng(2))
    assert not np.array_equal(first, second)


def test_daily_load_plot_is_saved(tmp_path):
    cell = Cell()
    path = tmp_path / "load.png"
    plot_daily_load(human_demand(cell, days=2, rng=np.random.default_rng(0)), cell, path)
    assert path.stat().st_size > 0


def test_humans_alone_rarely_congest_the_cell():
    """Tıkanıklığı ajanlar yaratmalı, insan trafiği tek başına değil."""
    cell = Cell()
    demand = human_demand(cell, days=100, rng=np.random.default_rng(0))
    assert cell.is_congested(demand).mean() < 0.02
