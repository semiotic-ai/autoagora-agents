# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from simulation import reward
from simulation.entity.entity import entitygroupfactory

from ..fixture import *


def test_traffic_reward(agentconfig):
    agentconfig["reward"] = [
        {"kind": "traffic", "multiplier": -1},
        {"kind": "traffic", "multiplier": 2},
    ]
    entities = {"indexer": entitygroupfactory(**agentconfig)}
    rew = reward.rewardfactory(rewards=agentconfig["reward"])
    assert isinstance(rew, reward.TrafficReward)
    traffic = np.random.rand(3)
    agent = entities["indexer"][0]
    agent.state.traffic = traffic
    # Mutipliers cancel out (-1 + 2 = 1)
    assert rew(agent=agent, entities=entities) == sum(traffic)  # type: ignore


def test_sumregretratio_reward(agentconfig, consumerconfig):
    agentconfig["reward"] = [
        {"kind": "sumregretratio", "multiplier": 1, "fromgroup": "consumer"},
    ]
    entities = {
        "indexer": entitygroupfactory(**agentconfig),
        "consumer": entitygroupfactory(**consumerconfig),
    }
    rew = reward.rewardfactory(rewards=agentconfig["reward"])
    assert isinstance(rew, reward.SumRegretRatio)
    traffic = np.random.rand(3)
    agent = entities["indexer"][0]
    agent.state.traffic = traffic
    assert rew(agent=agent, entities=entities) == sum(traffic) / 9  # type: ignore
