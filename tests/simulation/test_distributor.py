# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from simulation import distributor
from simulation.entity import entitygroupfactory

from ..fixture import *


def test_softmaxdistributor_one_indexer_all_traffic(agentconfig, consumerconfig):
    # Set up agents
    agentconfig["count"] = 2
    nproducts = 3
    indexers = entitygroupfactory(**agentconfig)
    indexers[0].state.value = np.zeros(nproducts)  # One agent's price is zero
    indexers[1].state.value = 5 * np.ones(nproducts)  # Other agent's price > budget
    consumers = entitygroupfactory(**consumerconfig)
    entities = {"consumer": consumers, "indexer": indexers}
    dist = distributor.distributorfactory(
        kind="softmax", source="consumer", to="indexer"
    )
    dist(entities=entities)
    assert sum(indexers[0].state.traffic) == 9
    assert sum(indexers[1].state.traffic) == 0
