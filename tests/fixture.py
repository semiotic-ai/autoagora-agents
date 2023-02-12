# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest


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
            "initial": np.ones(3),
        },
        "action": {
            "kind": "pricemultiplier",
            "low": np.zeros(3),
            "high": 3 * np.ones(3),
            "shape": (3,),
            "baseprice": 2 * np.ones(3),
        },
    }


@pytest.fixture
def consumerconfig():
    return {
        "kind": "entity",
        "count": 3,
        "group": "consumer",
        "state": {
            "kind": "budget",
            "low": np.zeros(3),
            "high": 3 * np.ones(3),
            "initial": np.ones(3),
            "traffic": np.ones(3),
        },
    }


@pytest.fixture
def simulationconfig():
    nproducts = 1
    return {
        "isa": {"kind": "softmax", "source": "consumer", "to": "indexer"},
        "entities": [
            {
                "kind": "entity",
                "count": 1,
                "group": "consumer",
                "state": {
                    "kind": "budget",
                    "low": 0,
                    "high": 1,
                    "initial": 0.5 * np.ones(nproducts),
                    "traffic": np.ones(nproducts),
                },
            },
            {
                "kind": "agent",
                "count": 2,
                "group": "indexer",
                "state": {
                    "kind": "price",
                    "low": np.zeros(nproducts),
                    "high": 3 * np.ones(nproducts),
                    "initial": np.ones(nproducts),
                },
                "action": {
                    "kind": "price",
                    "low": np.zeros(nproducts),
                    "high": 3 * np.ones(nproducts),
                    "shape": (nproducts,),
                },
                "reward": [
                    {
                        "kind": "traffic",
                        "multiplier": 1,
                    }
                ],
                "observation": [
                    {
                        "kind": "bandit",
                    }
                ],
            },
        ],
    }
