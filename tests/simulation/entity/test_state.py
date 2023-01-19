# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from simulation.entity import state


@pytest.fixture
def s():
    return state.State(low=0, high=3, initial=np.array([1, 2, 3]))


def test_state_init():
    a = np.array([1, 2, 3])
    s = state.State(low=0, high=3, initial=a)
    assert (s.space.low == np.zeros(3)).all()
    assert (s.space.high == 3 * np.ones(3)).all()
    assert (s.value == a).all()


def test_state_initial_oob():
    a = np.array([-1, -1, -1])
    s = state.State(low=0, high=3, initial=a)
    assert (s.value == np.array([0, 0, 0])).all()


def test_state_update(s):
    s.value = np.array([3, 3, 3])
    assert (s.value == np.array([3, 3, 3])).all()


def test_state_reset(s):
    s.state = np.array([3, 3, 3])
    s.reset()
    assert (s.value == np.array([1, 2, 3])).all()


def test_pricestate_factory():
    config = {
        "kind": "price",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "initial": np.zeros(3),
    }
    s = state.statefactory(**config)
    assert isinstance(s, state.PriceState)


def test_budgetstate_factory():
    config = {
        "kind": "budget",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "initial": np.zeros(3),
        "traffic": np.ones(3),
    }
    s = state.statefactory(**config)
    assert isinstance(s, state.BudgetState)
