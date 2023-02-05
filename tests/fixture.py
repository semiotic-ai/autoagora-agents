# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np


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
