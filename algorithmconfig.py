# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from autoagora_agents import algorithm_ingredient


@algorithm_ingredient.config
def config():
    seed = 0
    agents = [
        {
            "kind": "ppobandit",
            "group": "indexer",
            "count": 2,
            "bufferlength": 10,
            "actiondistribution": {
                "kind": "gaussian",
                "initial_mean": [1.0],
                "initial_stddev": [0.5],
                "minmean": [0.0],
                "maxmean": [2.0],
                "minstddev": [0.1],
                "maxstddev": [1.0],
            },
            "optimizer": {"kind": "sgd", "lr": 0.01},
            "ppoiterations": 2,
            "epsclip": 0.1,
            "entropycoeff": 1e-1,
            "pullbackstrength": 1,
            "stddevfallback": True,
        }
    ]
