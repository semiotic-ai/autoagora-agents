# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from simulation.entity import action


@pytest.fixture
def a():
    return action.Action(low=0, high=3, shape=(3,))


def test_action_init(a):
    assert (a.space.low == np.zeros(3)).all()
    assert (a.space.high == 3 * np.ones(3)).all()


def test_action_update(a):
    a.value = np.array([3, 3, 3])
    assert (a.value == np.array([3, 3, 3])).all()


def test_priceaction_factory():
    config = {
        "kind": "price",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "shape": (3,),
    }
    a = action.actionfactory(**config)
    assert isinstance(a, action.PriceAction)


def test_pricemultiplieraction_factory():
    config = {
        "kind": "pricemultiplier",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "shape": (3,),
        "baseprice": 2 * np.ones(3),
    }
    a = action.actionfactory(**config)
    assert isinstance(a, action.PriceMultiplierAction)


def test_budgetaction_factory():
    config = {
        "kind": "budget",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "shape": (3,),
    }
    s = action.actionfactory(**config)
    assert isinstance(s, action.BudgetAction)
