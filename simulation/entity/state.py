# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import gymnasium
import numpy as np
from numpy.typing import NDArray

import experiment


class State:
    """State of an entity.

    Attributes:
        low (float | NDArray): The lower bound of the state space.
        high (float | NDArray): The upper bound of the state space
        initial (NDArray): The initial value of the state
        value (NDArray): The state of the entity
        traffic (NDArray): Either how many queries the agent received in the last timestep
            or how many it sent out. Each element of the array is a different query type.
    """

    def __init__(
        self, *, low: float | NDArray, high: float | NDArray, initial: NDArray
    ) -> None:
        self.space = gymnasium.spaces.Box(low, high, shape=np.shape(initial))
        self.initial = initial
        self._state = np.zeros_like(initial)
        self.value = initial
        self.traffic = np.zeros_like(initial)

    def reset(self) -> None:
        """Reset the state."""
        self.value = self.initial
        self.traffic = np.zeros_like(self.initial)

    @property
    def value(self) -> NDArray:
        return self._state

    @value.setter
    def value(self, v: NDArray) -> None:
        v = experiment.applybounds(v, self.space.low, self.space.high)  # type: ignore
        self._state = v


class PriceState(State):
    """The price of each query type.

    Use "price" as the "kind" of state in the config.
    """

    def __init__(self, *, low: float, high: float, initial: NDArray) -> None:
        super().__init__(low=low, high=high, initial=initial)


class BudgetState(State):
    """The budget across all queries.

    The default budget in the studio is 0.003.

    Use "budget" as the "kind" of state in the config.

    Attributes:
        initialtraffic (NDArray): The initial value of the traffic vector.
    """

    def __init__(
        self, *, low: float, high: float, initial: NDArray, traffic: NDArray
    ) -> None:
        super().__init__(low=low, high=high, initial=initial)
        self.initialtraffic = traffic
        self.traffic = traffic

    def reset(self) -> None:
        """Reset the state."""
        self.value = self.initial
        self.traffic = self.initialtraffic


def statefactory(
    *, kind: str, low: float, high: float, initial: NDArray, **kwargs
) -> State:
    """Instantiate a new state.

    Keyword Arguments:
        kind (str): The type of state to instantiate.
            "price" -> PriceState
            "budget" -> BudgetState
        low (float): The lower bound of the state space
        high (float): The upper bound of the state space
        initial (NDArray): The initial value of the state.

    Returns:
        State: An instantiated state.
    """
    states = {"price": PriceState, "budget": BudgetState}
    return experiment.factory(
        kind, states, low=low, high=high, initial=initial, **kwargs
    )
