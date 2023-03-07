# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from autoagora_agents import algorithm_ingredient


@algorithm_ingredient.config
def config():
    agents = [
        {
            "kind": "ppobandit",
            "group": "indexer",
            "count": 1,
            "bufferlength": 10,
            "actiondistribution": {
                "kind": "gaussian",
                "initial_mean": [0.1],
                "initial_stddev": [0.1],
                "minmean": [0.0],
                "maxmean": [2.0],
                "minstddev": [1e-10],
                "maxstddev": [1.0],
            },
            "optimizer": {"kind": "sgd", "lr": 0.01},
            "ppoiterations": 2,
            "epsclip": 0.01,
            "entropycoeff": 1.0,
            "pullbackstrength": 0.0,
            "stddevfallback": True,
        }
    ]
