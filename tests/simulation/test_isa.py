# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from simulation import isa
from simulation.entity import entitygroupfactory

from ..fixture import *


def test_sigmoidisa_one_indexer_all_traffic(agentconfig, consumerconfig):
    # Set up agents
    agentconfig["count"] = 2
    nproducts = 3
    indexers = entitygroupfactory(**agentconfig)
    indexers[0].state.value = np.zeros(nproducts)  # One agent's price is zero
    indexers[1].state.value = 5 * np.ones(nproducts)  # Other agent's price > budget
    consumers = entitygroupfactory(**consumerconfig)
    entities = {"consumer": consumers, "indexer": indexers}
    _isa = isa.isafactory(kind="softmax", source="consumer", to="indexer")
    _isa(entities=entities)
    assert sum(indexers[0].state.traffic) == 9
    assert sum(indexers[1].state.traffic) == 0
