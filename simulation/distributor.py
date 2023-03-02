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
    def softmaxmask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute the columnwise softmax for elements that are True in the mask.

        Arguments:
            x (np.ndarray): The array for which to compute the softmax.
            mask (np.ndarray): The mask array. The function works for True values.

        Returns:
            np.ndarray: An array in which the indices that are True in the mask are
                columnwise softmaxed, and the indices that are False are zeroed.
        """
        x = np.atleast_2d(x)
        mask = np.atleast_2d(mask)
        y = np.zeros_like(x)
        # Iterate over columns
        for j in range(x.shape[1]):
            # Get masked column
            xmask = x[:, j][mask[:, j]]
            if xmask.size <= 0:
                continue
            # Subtract max for numerical stability as np.exp(inf) is inf
            # but np.exp(-inf) is 0
            ex = np.exp(xmask - np.max(xmask))
            # Set value into masked indices
            y[:, j][mask[:, j]] = ex / np.sum(ex)
        return y

    def __call__(self, *, entities: dict[str, list[Entity]]) -> None:
        source = entities[self.source]
        to = entities[self.to]
        prices = np.atleast_2d(np.vstack([t.state.value for t in to]))
        traffics = np.zeros_like(prices)
        for s in source:
            budget = s.state.value
            # If above budget, don't get any traffic
            mask = prices <= budget
            # Compute how much traffic goes to each agent below the budget
            percenttraffic = self.softmaxmask(prices, mask)
            # Accumulate traffic values per agent
            traffics += np.multiply(percenttraffic, s.state.traffic)

        for traffic, t in zip(traffics, to):
            t.state.traffic = traffic


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
