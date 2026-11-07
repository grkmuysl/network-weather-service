"""Tahmin modeli testleri."""

import numpy as np
import pytest

from nwl.evaluate_forecasts import score_model, synthetic_series
from nwl.forecast import (
    SLOTS_PER_DAY,
    DeseasonalizedArima,
    MovingAverage,
    SameTimeLastWeek,
    SameTimeYesterday,
    SmallLstm,
)
from nwl.protocol import HORIZON_SLOTS


def test_moving_average_is_flat_at_the_recent_mean():
    history = np.array([9.0, 9.0, 1.0, 3.0])
    assert MovingAverage(window=2).forecast(history).tolist() == [2.0] * HORIZON_SLOTS


def test_same_time_yesterday_repeats_the_last_day():
    history = np.arange(3.0 * SLOTS_PER_DAY)
    assert np.array_equal(SameTimeYesterday().forecast(history), history[-SLOTS_PER_DAY:])


def test_same_time_last_week_repeats_the_matching_day():
    history = np.arange(8.0 * SLOTS_PER_DAY)
    expected = history[SLOTS_PER_DAY : 2 * SLOTS_PER_DAY]
    assert np.array_equal(SameTimeLastWeek().forecast(history), expected)


def test_perfect_forecast_scores_zero():
    series = np.tile(np.sin(np.linspace(0, 6, SLOTS_PER_DAY)), 4)
    score = score_model(SameTimeYesterday(), series, test_start=2 * SLOTS_PER_DAY)
    assert score.mae == pytest.approx(0)
    assert score.rmse == pytest.approx(0)


def test_score_reports_known_error():
    series = np.zeros(3 * SLOTS_PER_DAY)
    series[-SLOTS_PER_DAY:] = 2.0
    score = score_model(SameTimeYesterday(), series, test_start=2 * SLOTS_PER_DAY)
    assert score.mae == 2.0
    assert score.rmse == 2.0


def test_arima_follows_the_daily_cycle_better_than_a_flat_line():
    series = synthetic_series(days=14, seed=0)
    test_start = 10 * SLOTS_PER_DAY
    arima = DeseasonalizedArima()
    arima.fit(series[:test_start])
    arima_score = score_model(arima, series, test_start)
    flat_score = score_model(MovingAverage(), series, test_start)
    assert arima_score.mae < flat_score.mae / 2


def test_lstm_learns_a_repeating_day():
    series = synthetic_series(days=14, seed=0)
    test_start = 10 * SLOTS_PER_DAY
    lstm = SmallLstm(epochs=15)
    lstm.fit(series[:test_start])
    assert lstm.forecast(series[:test_start]).shape == (HORIZON_SLOTS,)
    lstm_score = score_model(lstm, series, test_start)
    assert lstm_score.mae < score_model(MovingAverage(), series, test_start).mae
