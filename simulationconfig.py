# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from simulation import simulation_ingredient


@simulation_ingredient.config
def config():
    nproducts = 1
    ntimesteps = 10000
    nepisodes = 1
    distributor = {"kind": "softmax", "source": "consumer", "to": "indexer"}
    entities = [
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
            "count": 1,
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
    ]
