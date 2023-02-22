# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from simulation import observation
from simulation.entity.entity import entitygroupfactory

from ..fixture import *


def test_bandit_observation(agentconfig):
    agentconfig["observation"] = [
        {"kind": "bandit"},
    ]
    entities = {"indexer": entitygroupfactory(**agentconfig)}
    obs = observation.observationfactory(observations=agentconfig["observation"])
    assert isinstance(obs, observation.BanditObservation)
    agent = entities["indexer"][0]
    assert np.allclose(obs(agent=agent, entities=entities), np.array([]))  # type: ignore
