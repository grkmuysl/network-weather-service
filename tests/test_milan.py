"""Milano verisi dönüşüm testleri. Gerçek veri dosyası gerekmez."""

import numpy as np
import pytest

from nwl.milan import (
    REFERENCE_LOAD,
    SQUARES,
    STEPS_PER_DAY,
    choose_cells,
    sum_blocks,
    to_load,
    to_slots,
)


def test_blocks_keep_the_total_and_group_neighbours():
    by_square = np.zeros((SQUARES, 2))
    by_square[[0, 4, 404], 0] = 1.0  # aynı blok
    by_square[5, 1] = 7.0  # sağdaki blok
    blocks = sum_blocks(by_square)
    assert blocks.shape == (400, 2)
    assert blocks[0].tolist() == [3.0, 0.0]
    assert blocks[1].tolist() == [0.0, 7.0]


def test_slots_keep_the_average():
    steps = np.array([3.0, 3.0, 9.0, 9.0, 0.0, 6.0])
    slots = to_slots(steps)
    assert slots == pytest.approx([3.0, 7.0, 6.0, 4.0])
    assert slots.mean() == pytest.approx(steps.mean())


def test_load_is_scaled_from_the_training_period_only():
    slots = np.array([1.0, 1.0, 1.0, 100.0])
    assert to_load(slots, train_slots=3)[0] == pytest.approx(REFERENCE_LOAD)


def test_cells_are_chosen_by_their_rules():
    blocks = np.ones((8, 7 * STEPS_PER_DAY))
    blocks *= np.arange(1, 9)[:, None]
    hours = np.arange(blocks.shape[1]) / 6 % 24
    evening = (hours >= 20) & (hours < 23)
    blocks[5, evening] *= 3
    chosen = choose_cells(blocks, 7 * STEPS_PER_DAY)
    assert chosen["Akşam yoğun"] == 5
    assert chosen["Merkez"] == 7
