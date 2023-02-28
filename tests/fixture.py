# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from simulation import environment


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
        "ntimesteps": 2,
        "nepisodes": 1,
        "distributor": {"kind": "softmax", "source": "consumer", "to": "indexer"},
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


@pytest.fixture
def env(simulationconfig):
    return environment(
        simulationconfig["distributor"],
        simulationconfig["entities"],
        simulationconfig["ntimesteps"],
        simulationconfig["nepisodes"],
    )


@pytest.fixture
def gaussianconfig():
    return {
        "kind": "gaussian",
        "initial_mean": [1.0],
        "initial_stddev": [0.5],
        "minmean": [0.0],
        "maxmean": [2.0],
        "minstddev": [0.1],
        "maxstddev": [1.0],
    }


@pytest.fixture
def degenerateconfig():
    return {
        "kind": "degenerate",
        "initial_value": [1.0],
        "minvalue": [0.0],
        "maxvalue": [2.0],
    }


@pytest.fixture
def scaledgaussianconfig():
    return {
        "kind": "scaledgaussian",
        "initial_mean": [1.0],
        "initial_stddev": [1.0],
        "minmean": [1.0],
        "maxmean": [5.0],
        "minstddev": [0.1],
        "maxstddev": [1.0],
        "scalefactor": [1.0],
    }


@pytest.fixture
def predeterminedconfig():
    return {
        "kind": "predetermined",
        "group": "indexer",
        "count": 1,
        "timestamps": [0, 3, 6],
        "vals": [np.zeros(1), np.ones(1), 2 * np.ones(1)],
    }


@pytest.fixture
def vpgbanditconfig():
    return {
        "kind": "vpgbandit",
        "group": "indexer",
        "count": 1,
        "bufferlength": 2,
        "actiondistribution": {
            "kind": "gaussian",
            "initial_mean": [1.0, 1.0, 1.0],
            "initial_stddev": [0.1, 0.1, 0.1],
            "minmean": [0.0, 0.0, 0.0],
            "maxmean": [2.0, 2.0, 2.0],
            "minstddev": [0.1, 0.1, 0.1],
            "maxstddev": [1.0, 1.0, 1.0],
        },
        "optimizer": {"kind": "sgd", "lr": 0.001},
    }


@pytest.fixture
def ppobanditconfig():
    return {
        "kind": "ppobandit",
        "group": "indexer",
        "count": 1,
        "bufferlength": 2,
        "actiondistribution": {
            "kind": "gaussian",
            "initial_mean": [1.0, 1.0, 1.0],
            "initial_stddev": [0.1, 0.1, 0.1],
            "minmean": [0.0, 0.0, 0.0],
            "maxmean": [2.0, 2.0, 2.0],
            "minstddev": [0.1, 0.1, 0.1],
            "maxstddev": [1.0, 1.0, 1.0],
        },
        "optimizer": {"kind": "sgd", "lr": 0.001},
        "ppoiterations": 2,
        "epsclip": 0.1,
        "entropycoeff": 1e-1,
        "pullbackstrength": 1,
        "stddevfallback": True,
    }


@pytest.fixture
def rmppobanditconfig():
    return {
        "kind": "rmppobandit",
        "group": "indexer",
        "count": 1,
        "bufferlength": 2,
        "actiondistribution": {
            "kind": "gaussian",
            "initial_mean": [1.0, 1.0, 1.0],
            "initial_stddev": [0.1, 0.1, 0.1],
            "minmean": [0.0, 0.0, 0.0],
            "maxmean": [2.0, 2.0, 2.0],
            "minstddev": [0.1, 0.1, 0.1],
            "maxstddev": [1.0, 1.0, 1.0],
        },
        "optimizer": {"kind": "sgd", "lr": 0.001},
        "ppoiterations": 2,
        "epsclip": 0.1,
        "entropycoeff": 1e-1,
        "pullbackstrength": 1,
        "stddevfallback": True,
    }
