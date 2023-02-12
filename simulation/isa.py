# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import numpy as np
from jax.nn import softmax

import experiment
from simulation.entity import Entity


class ISA(ABC):
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


class SoftmaxISA(ISA):
    """Allocates traffic via a softmax function.

    However, if an indexer's price exceeds a consumer's budget, the indexer gets 0 traffic.
    """

    def __init__(self, *, source: str, to: str) -> None:
        super().__init__(source=source, to=to)

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
                # if price > budget, set to -np.inf
                ps = np.array([b - p if p <= b else -np.inf for p in prices])
                # Run softmax to see how much of the traffic goes to each indexer
                allocs[j, :] = t * softmax(ps)

            # Get total traffic per indexer for product i
            ttraffics[:, i] = np.sum(allocs, axis=0)

        # Assign computed value back to state
        for i, t in enumerate(to):
            t.state.traffic = ttraffics[i, :]


def isafactory(*, kind: str, source: str, to: str, **kwargs) -> ISA:
    """Instantiate a new ISA.

    Keyword Arguments:
        kind (str): The type of ISA to instantiate.
            "softmax" -> SoftmaxISA
        source (str): The group from which the query comes. E.g., "consumer"
        to (str): The group to which the query goes. E.g., "indexer"

    Returns:
        ISA: An instantiated ISA.
    """
    isas = {"softmax": SoftmaxISA}
    return experiment.factory(kind, isas, source=source, to=to, **kwargs)