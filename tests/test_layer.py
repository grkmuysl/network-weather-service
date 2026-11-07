"""Yayın protokolü, NWL ve zamanlayıcı testleri."""

import numpy as np
import pytest

from nwl.agents import SECONDS_PER_STEP, TASK_RATE_MBPS, Tasks, TaskSettings, generate_tasks
from nwl.base_station import (
    BaseStation,
    LoadMeter,
    ModelForecaster,
    NoisyForecaster,
    PerfectForecaster,
)
from nwl.cell import Cell
from nwl.forecast import MovingAverage, SameTimeLastWeek, SameTimeYesterday
from nwl.layer import (
    SAFETY_MARGIN_MINUTES,
    NetworkWeatherLayer,
    StartOption,
    quietest_slot,
    random_in_quiet_window,
    schedule_tasks,
)
from nwl.metrics import compute_metrics
from nwl.protocol import (
    HORIZON_SLOTS,
    SLOT_MINUTES,
    ForecastEntry,
    ForecastMessage,
    broadcast_bits_per_second,
    decode,
    encode,
)
from nwl.scenarios import (
    MARGIN_DAYS,
    ForecastError,
    World,
    measured_past,
    run_scenario,
    run_world,
)
from nwl.traffic import STEPS_PER_DAY, human_demand


def station_with_loads(loads: list[float]) -> BaseStation:
    """Her slotun yükü verilen değere eşit olan baz istasyonu."""
    by_minute = np.repeat(np.array(loads), SLOT_MINUTES)
    return BaseStation(PerfectForecaster(by_minute), total_slots=len(loads))


def deferrable_task(arrival: int, deadline: float, minutes: float = 5.0) -> Tasks:
    return Tasks(
        arrival=np.array([arrival]),
        size_mb=np.array([minutes * TASK_RATE_MBPS * SECONDS_PER_STEP]),
        urgent=np.array([False]),
        deadline=np.array([deadline]),
    )


def test_message_survives_encoding():
    entries = (ForecastEntry(3, 0.25, 0.9), ForecastEntry(4, 1.2, 0.5))
    message = ForecastMessage(45, entries)
    assert decode(encode(message)) == message


def test_truncated_message_is_rejected():
    message = ForecastMessage(0, (ForecastEntry(0, 0.5, 1.0), ForecastEntry(1, 0.5, 1.0)))
    with pytest.raises(ValueError):
        decode(encode(message)[:-5])


def test_full_broadcast_size_and_overhead():
    loads = [0.5] * HORIZON_SLOTS
    assert len(station_with_loads(loads).broadcast(0)) == 6 + 5 * HORIZON_SLOTS
    assert broadcast_bits_per_second() == pytest.approx(486 * 8 / 900)


def test_perfect_forecast_is_the_slot_average():
    by_minute = np.concatenate([np.full(SLOT_MINUTES, 0.2), np.full(SLOT_MINUTES, 0.4)])
    loads, confidence = PerfectForecaster(by_minute).predict(0, np.array([0, 1]))
    assert loads.tolist() == pytest.approx([0.2, 0.4])
    assert confidence.tolist() == [1.0, 1.0]


def test_broadcast_starts_at_the_current_slot_and_stops_at_the_end():
    message = decode(station_with_loads([0.1, 0.2, 0.3]).broadcast(20))
    assert message.issued_minute == SLOT_MINUTES
    assert [entry.time_slot for entry in message.entries] == [1, 2]


def test_options_stay_between_now_and_latest_start():
    layer = NetworkWeatherLayer()
    layer.receive(station_with_loads([0.1, 0.2, 0.3]).broadcast(0))
    options = layer.start_options(now=10, latest_start=20)
    assert [(o.first_minute, o.last_minute) for o in options] == [(10, 14), (15, 20)]


def test_quietest_slot_picks_the_lowest_forecast():
    options = [StartOption(0, 14, 0.5), StartOption(15, 29, 0.1), StartOption(30, 44, 0.3)]
    assert quietest_slot(options, np.random.default_rng(0)) == 15


def test_random_choice_stays_inside_the_quiet_window():
    options = [StartOption(0, 14, 0.5), StartOption(15, 29, 0.10), StartOption(30, 44, 0.12)]
    rng = np.random.default_rng(0)
    picks = {random_in_quiet_window(options, rng) for _ in range(300)}
    assert min(picks) >= 15
    assert max(picks) <= 44
    assert len(picks) > 20


def test_urgent_tasks_are_never_moved():
    tasks = generate_tasks(Cell(), days=2, rng=np.random.default_rng(0))
    station = BaseStation(
        PerfectForecaster(np.full(2 * STEPS_PER_DAY, 0.5)), total_slots=2 * 96
    )
    starts = schedule_tasks(tasks, station, quietest_slot, np.random.default_rng(0))
    assert np.array_equal(starts[tasks.urgent], tasks.arrival[tasks.urgent])


def test_task_moves_to_the_quiet_slot_before_its_deadline():
    station = station_with_loads([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.05, 0.9])
    task = deferrable_task(arrival=0, deadline=SAFETY_MARGIN_MINUTES + 5 + 44)
    starts = schedule_tasks(task, station, quietest_slot, np.random.default_rng(0))
    assert starts[0] == 30


def test_task_without_spare_time_starts_immediately():
    station = station_with_loads([0.9, 0.1])
    task = deferrable_task(arrival=3, deadline=10.0)
    starts = schedule_tasks(task, station, quietest_slot, np.random.default_rng(0))
    assert starts[0] == 3


def measured(scenario: str):
    cell = Cell()
    runs = [
        compute_metrics(run_scenario(scenario, cell, days=7, seed=seed), cell, MARGIN_DAYS)
        for seed in range(3)
    ]
    return runs


def test_scenario_simulates_margin_days_around_the_measured_days():
    result = run_scenario("a", Cell(), days=2, seed=0)
    assert len(result.human_mbps) == (2 + 2 * MARGIN_DAYS) * STEPS_PER_DAY


def test_all_agent_scenarios_share_the_same_tasks():
    b = run_scenario("b", Cell(), days=2, seed=1)
    d = run_scenario("d", Cell(), days=2, seed=1)
    assert np.array_equal(b.tasks.arrival, d.tasks.arrival)


def test_full_nwl_lowers_the_peak_compared_with_ignoring_the_forecast():
    b, d = measured("b"), measured("d")
    assert np.mean([r.congested_minutes_per_day for r in d]) < np.mean(
        [r.congested_minutes_per_day for r in b]
    )


def test_randomisation_removes_the_herd_peak():
    c, d = measured("c"), measured("d")
    assert np.mean([r.peak_load for r in d]) < np.mean([r.peak_load for r in c])


def test_meter_adds_task_load_with_a_partial_last_minute():
    cell = Cell(capacity_mbps=100)
    meter = LoadMeter(np.zeros(5), cell)
    meter.add_task(1, 1.5)
    assert meter.load == pytest.approx([0, 0.02, 0.01, 0, 0])


def test_low_confidence_makes_a_slot_look_busier():
    payload = encode(
        ForecastMessage(0, (ForecastEntry(0, 0.2, 0.5), ForecastEntry(1, 0.3, 1.0)))
    )
    layer = NetworkWeatherLayer()
    layer.receive(payload)
    options = layer.start_options(now=0, latest_start=29)
    assert [o.cautious_load for o in options] == pytest.approx([0.45, 0.3])
    assert quietest_slot(options, np.random.default_rng(0)) == 15


def world_with_flat_history() -> tuple[np.ndarray, LoadMeter]:
    history = np.full(10 * HORIZON_SLOTS, 0.4)
    return history, LoadMeter(np.full(2 * STEPS_PER_DAY, 0.4), Cell())


def test_model_forecaster_sees_only_finished_slots():
    history, meter = world_with_flat_history()
    forecaster = ModelForecaster(MovingAverage(window=1), history, meter)
    meter.load[:SLOT_MINUTES] = 0.9
    meter.load[SLOT_MINUTES:] = 0.1
    loads, _ = forecaster.predict(1, np.array([1, 2]))
    assert loads.tolist() == pytest.approx([0.9, 0.9])


def test_confidence_falls_when_the_model_starts_missing():
    history, meter = world_with_flat_history()
    forecaster = ModelForecaster(SameTimeYesterday(), history, meter)
    slots = np.array([0])
    _, before = forecaster.predict(0, slots)
    assert before[0] == pytest.approx(1.0)

    meter.load[:] = 0.9
    forecaster.predict(4, np.array([4]))
    _, after = forecaster.predict(8, np.array([8]))
    assert after[0] < 1.0


def test_scheduling_records_every_task_in_the_meter():
    cell = Cell()
    tasks = generate_tasks(cell, 2, np.random.default_rng(0))
    meter = LoadMeter(np.zeros(2 * STEPS_PER_DAY), cell)
    station = BaseStation(PerfectForecaster(np.full(2 * STEPS_PER_DAY, 0.5)), total_slots=192)
    schedule_tasks(tasks, station, quietest_slot, np.random.default_rng(0), meter)
    recorded = meter.load.sum() * cell.capacity_mbps * SECONDS_PER_STEP
    assert recorded == pytest.approx(tasks.size_mb.sum(), rel=0.02)


def test_world_runs_with_a_model_in_the_loop():
    cell = Cell()
    human = human_demand(cell, 11, np.random.default_rng(0))
    world = World("deneme", human, start_day=8, days=3)
    past = measured_past(world, cell)
    model = SameTimeLastWeek()
    result = run_world(world, cell, "d", seed=0, model=model, past=past)
    metrics = compute_metrics(result, cell, MARGIN_DAYS)
    assert metrics.deadline_violation_percent < 5


def test_noise_changes_the_schedule_but_not_the_tasks():
    cell = Cell()
    human = human_demand(cell, 11, np.random.default_rng(0))
    world = World("deneme", human, start_day=8, days=3)
    clean = run_world(world, cell, "d", seed=0)
    noisy = run_world(world, cell, "d", seed=0, error=ForecastError(noise=0.3))
    assert np.array_equal(clean.tasks.arrival, noisy.tasks.arrival)
    assert not np.array_equal(clean.agent_mbps, noisy.agent_mbps)


def test_forecast_error_stays_the_same_across_broadcasts():
    perfect = PerfectForecaster(np.full(10 * SLOT_MINUTES, 0.5))
    noisy = NoisyForecaster(perfect, 0.1, 0.9, 10, np.random.default_rng(0))
    first, _ = noisy.predict(0, np.array([3, 4]))
    second, _ = noisy.predict(2, np.array([3, 4]))
    assert first.tolist() == second.tolist()


def test_memory_makes_neighbouring_errors_alike():
    perfect = PerfectForecaster(np.full(5000 * SLOT_MINUTES, 0.5))
    rng = np.random.default_rng(0)
    noisy = NoisyForecaster(perfect, 0.1, 0.95, 5000, rng)
    assert np.corrcoef(noisy.errors[:-1], noisy.errors[1:])[0, 1] > 0.9
    assert noisy.errors.std() == pytest.approx(0.1, rel=0.2)


def test_task_settings_change_the_share_of_urgent_tasks():
    rng = np.random.default_rng(0)
    tasks = generate_tasks(Cell(), 3, rng, settings=TaskSettings(urgent_share=0.9))
    assert tasks.urgent.mean() == pytest.approx(0.9, abs=0.02)


def test_zero_margin_stays_in_the_lowest_slot():
    options = [StartOption(0, 14, 0.10), StartOption(15, 29, 0.11)]
    rng = np.random.default_rng(0)
    picks = {random_in_quiet_window(options, rng, quiet_margin=0.0) for _ in range(100)}
    assert max(picks) <= 14


def test_uncertain_forecast_widens_the_quiet_window():
    options = [StartOption(0, 14, 0.2, 0.1), StartOption(15, 29, 0.35, 0.1)]
    rng = np.random.default_rng(0)
    narrow = {random_in_quiet_window(options, rng, error_weight=0.0) for _ in range(200)}
    wide = {random_in_quiet_window(options, rng, error_weight=2.0) for _ in range(200)}
    assert max(narrow) <= 14
    assert max(wide) >= 15


def test_noisy_forecaster_reports_its_expected_error():
    perfect = PerfectForecaster(np.full(10 * SLOT_MINUTES, 0.5))
    noisy = NoisyForecaster(perfect, 0.1, 0.0, 10, np.random.default_rng(0))
    _, confidence = noisy.predict(0, np.array([0, 1]))
    assert confidence == pytest.approx(1 - 0.0798 / 0.5, abs=0.001)
