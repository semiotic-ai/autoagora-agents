# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from simulation import distributor
from simulation.entity import entitygroupfactory

from ..fixture import *


def test_softmaxdistributor_softmaxmask_degenerate():
    # All values are False in the mask
    # Should return zeros
    x = np.ones((3, 2))
    mask = np.full((3, 2), False)
    dist = distributor.distributorfactory(
        kind="softmax", source="consumer", to="indexer"
    )
    y = dist.softmaxmask(x, mask)  # type: ignore
    assert np.array_equal(y, np.zeros_like(x))


def test_softmaxdistributor_softmaxmask_partial_mask():
    x = np.ones((3, 2))
    mask = np.full((3, 2), False)
    mask[0] = [True, True]
    dist = distributor.distributorfactory(
        kind="softmax", source="consumer", to="indexer"
    )
    y = dist.softmaxmask(x, mask)  # type: ignore
    expected = np.zeros_like(x)
    expected[0] = [1.0, 1.0]
    assert np.array_equal(y, expected)


def test_softmaxdistributor_softmaxmask_no_mask():
    x = np.ones((2, 2))
    mask = np.full((2, 2), True)
    dist = distributor.distributorfactory(
        kind="softmax", source="consumer", to="indexer"
    )
    y = dist.softmaxmask(x, mask)  # type: ignore
    expected = 0.5 * np.ones_like(x)
    assert np.array_equal(y, expected)


def test_softmaxdistributor_softmaxmask_different_masks_per_column():
    x = np.ones((2, 2))
    mask = np.full((2, 2), True)
    mask[1, 1] = False
    dist = distributor.distributorfactory(
        kind="softmax", source="consumer", to="indexer"
    )
    y = dist.softmaxmask(x, mask)  # type: ignore
    expected = 0.5 * np.ones_like(x)
    expected[:, 1] = [1.0, 0.0]
    assert np.array_equal(y, expected)


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


def test_softmaxdistributor_all_indexers_over_budget(agentconfig, consumerconfig):
    # Set up agents
    agentconfig["count"] = 2
    nproducts = 3
    indexers = entitygroupfactory(**agentconfig)
    # Both agents over budget
    indexers[0].state.value = 5 * np.ones(nproducts)
    indexers[1].state.value = 5 * np.ones(nproducts)
    consumers = entitygroupfactory(**consumerconfig)
    entities = {"consumer": consumers, "indexer": indexers}
    dist = distributor.distributorfactory(
        kind="softmax", source="consumer", to="indexer"
    )
    dist(entities=entities)
    assert sum(indexers[0].state.traffic) == 0
    assert sum(indexers[1].state.traffic) == 0
