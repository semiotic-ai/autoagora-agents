# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from simulation.entity import Agent, Entity, entity
from simulation.entity.action import *
from simulation.entity.state import *


@pytest.fixture
def entityconfig():
    return {
        "kind": "entity",
        "count": 5,
        "group": "consumer",
        "state": {
            "kind": "budget",
            "low": np.zeros(3),
            "high": 3 * np.ones(3),
            "initial": np.zeros(3),
            "traffic": np.ones(3),
        },
    }


@pytest.fixture
def agentconfig():
    return {
        "kind": "agent",
        "count": 7,
        "group": "indexer",
        "state": {
            "kind": "price",
            "low": np.zeros(3),
            "high": 3 * np.ones(3),
            "initial": np.zeros(3),
        },
        "action": {
            "kind": "pricemultiplier",
            "low": np.zeros(3),
            "high": 3 * np.ones(3),
            "shape": (3,),
            "baseprice": 2 * np.ones(3),
        },
    }


def test_entity_init(entityconfig):
    es = entity.entitygroupfactory(**entityconfig)
    assert isinstance(es[0], Entity)
    assert len(es) == 5
    assert isinstance(es[0].state, BudgetState)


def test_agent_init(agentconfig):
    ags = entity.entitygroupfactory(**agentconfig)
    assert isinstance(ags[0], Agent)
    assert len(ags) == 7
    assert isinstance(ags[0].state, PriceState)
    assert isinstance(ags[0].action, PriceMultiplierAction)
