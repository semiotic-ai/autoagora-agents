# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import numpy as np

import experiment
from simulation.entity import Entity


class Distributor(ABC):
    """The indexer selection algorithm base class.

    Attributes:
        source (str): The group from which the query comes. E.g., "consumer"
        to (str): The group to which the query goes. E.g., "indexer"
    """

    def __init__(self, *, source: str, to: str) -> None:
        super().__init__()
        self.source = source
        self.to = to

    @abstractmethod
    def __call__(self, *, entities: dict[str, list[Entity]]) -> None:
        """Choose how to allocate traffic from `source` to `to`.

        Keyword Arguments:
            entities (dict[str, list[Entity]]): A mapping from group names to entities
                in said group.
        """
        pass


class SoftmaxDistributor(Distributor):
    """Allocates traffic via a softmax function.

    However, if an indexer's price exceeds a consumer's budget, the indexer gets 0 traffic.

    Attributes:
        minprice (float): A large, negative price so as to ensure an indexer doesn't receive
            traffic.
    """

    def __init__(self, *, source: str, to: str) -> None:
        super().__init__(source=source, to=to)
        self.minprice = -1e20

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """np.ndarray: Compute softmax of input array."""
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex, axis=0)

    # FIXME: Clean up
    def __call__(self, *, entities: dict[str, list[Entity]]) -> None:
        source = entities[self.source]
        to = entities[self.to]
        nproducts = len(to[0].state.value)  # first
        nto = len(to)
        ttraffics = np.zeros((nto, nproducts))
        # For each product
        for i in range(nproducts):
            # Get all prices
            prices = [t.state.value[i] for t in to]  # nto
            # For each budget
            budgets = [s.state.value[i] for s in source]  # nsource
            straffics = [s.state.traffic[i] for s in source]  # nsource
            allocs = np.zeros((nproducts, nto))
            for j, (t, b) in enumerate(zip(straffics, budgets)):
                # if price > budget, set to -1e20
                ps = np.array([b - p if p <= b else self.minprice for p in prices])
                # Run softmax to see how much of the traffic goes to each indexer
                allocs[j, :] = t * self.softmax(ps)

            # Get total traffic per indexer for product i
            ttraffics[:, i] = np.sum(allocs, axis=0)

        # Assign computed value back to state
        for i, t in enumerate(to):
            t.state.traffic = ttraffics[i, :]


def distributorfactory(*, kind: str, source: str, to: str, **kwargs) -> Distributor:
    """Instantiate a new Distributor.

    Keyword Arguments:
        kind (str): The type of Distributor to instantiate.
            "softmax" -> SoftmaxDistributor
        source (str): The group from which the query comes. E.g., "consumer"
        to (str): The group to which the query goes. E.g., "indexer"

    Returns:
        Distributor: An instantiated Distributor.
    """
    distributors = {"softmax": SoftmaxDistributor}
    return experiment.factory(kind, distributors, source=source, to=to, **kwargs)
