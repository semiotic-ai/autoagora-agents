# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from simulation.entity.state import *
from simulation.entity.action import *
from simulation.dynamics import *


def test_pricestate_priceaction_dynamics():
    state = {
        "kind": "price",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "initial": np.zeros(3),
    }
    s = statefactory(**state)
    action = {
        "kind": "price",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "shape": (3,),
    }
    a = actionfactory(**action)
    a.action = np.array([1, 2, 3])
    dynamics(s, a)  # type: ignore
    assert (s.state == np.array([1, 2, 3])).all()


def test_pricestate_pricemultiplieraction_dynamics():
    state = {
        "kind": "price",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "initial": np.zeros(3),
    }
    s = statefactory(**state)
    action = {
        "kind": "pricemultiplier",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "shape": (3,),
        "baseprice": 0.1 * np.ones(3),
    }
    a = actionfactory(**action)
    a.action = np.array([1, 2, 3])
    dynamics(s, a)  # type: ignore
    assert np.allclose(s.state, np.array([0.1, 0.2, 0.3]))


def test_budgetstate_budgetaction_dynamics():
    state = {
        "kind": "budget",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "initial": np.zeros(3),
    }
    s = statefactory(**state)
    action = {
        "kind": "budget",
        "low": np.zeros(3),
        "high": 3 * np.ones(3),
        "shape": (3,),
    }
    a = actionfactory(**action)
    a.action = np.array([1, 2, 3])
    dynamics(s, a)  # type: ignore
    assert (s.state == np.array([1, 2, 3])).all()
