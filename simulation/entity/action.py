# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import gymnasium
from numpy.typing import NDArray

import experiment


class Action:
    """Action of an entity.

    Attributes:
        low (float | NDArray): The lower bound of the action space
        high (float | NDArray): The upper bound of the action space
        shape (tuple[int, ...]): The shape of the action space
        action (NDArray): The action of the entity
    """

    def __init__(
        self, *, low: float | NDArray, high: float | NDArray, shape: tuple[int, ...]
    ) -> None:
        self.space = gymnasium.spaces.Box(low, high, shape=shape)
        self._action = self.space.sample()

    @property
    def action(self) -> NDArray:
        return self._action

    @action.setter
    def action(self, v: NDArray) -> None:
        v = experiment.applybounds(v, self.space.low, self.space.high)  # type: ignore
        self._action = v


class PriceAction(Action):
    """The price of each query type.

    Use "price" as the "kind" of action in the config.
    """

    def __init__(self, *, low: float, high: float, shape: tuple[int, ...]) -> None:
        super().__init__(low=low, high=high, shape=shape)


class PriceMultiplierAction(Action):
    """The price multiplier and base price of each query type.

    Use "pricemultiplier" as the "kind" of action in the config.
    """

    def __init__(self, *, low: float, high: float, shape: tuple[int, ...]) -> None:
        super().__init__(low=low, high=high, shape=shape)

    @property
    def action(self) -> NDArray:
        return self._action

    @action.setter
    def action(self, v: NDArray) -> None:
        v = experiment.applybounds(v, self.space.low, self.space.high)  # type: ignore
        self._action = v


class BudgetAction(Action):
    """The budget across all queries.

    Use "budget" as the "kind" of action in the config.
    """

    def __init__(self, *, low: float, high: float, shape: tuple[int, ...]) -> None:
        super().__init__(low=low, high=high, shape=shape)


def actionfactory(
    *, kind: str, low: float, high: float, shape: tuple[int, ...]
) -> Action:
    """Instantiate a new action.

    Keyword Arguments:
        kind (str): The type of action to instantiate.
            "price" -> PriceAction
            "pricemultiplier" -> PriceMultiplierAction
            "budget" -> BudgetAction
        low (float): The lower bound of the action space
        high (float): The upper bound of the action space
        shape (tuple[int, ...]): The shape of the action space

    Returns:
        Action: An instantiated action.
    """
    states = {
        "price": PriceAction,
        "budget": BudgetAction,
        "pricemultiplier": PriceMultiplierAction,
    }
    return experiment.factory(kind, states, low=low, high=high, shape=shape)
