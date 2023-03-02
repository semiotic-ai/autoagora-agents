# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import gymnasium
import numpy as np

import experiment


class Action:
    """Action of an entity.

    Keyword Arguments:
        low (float | np.ndarray): The lower bound of the action space
        high (float | np.ndarray): The upper bound of the action space
        shape (tuple[int, ...]): The shape of the action space
        seed (int): The seed of the random action generator.

    Attributes:
        space (gymnasium.spaces.Box): The action space.
    """

    def __init__(
        self,
        *,
        low: float | np.ndarray,
        high: float | np.ndarray,
        shape: tuple[int, ...],
        seed: int
    ) -> None:
        self.space = gymnasium.spaces.Box(low, high, shape=shape, seed=seed)
        self._action = self.space.sample()

    @property
    def value(self) -> np.ndarray:
        """np.ndarray: The action of the entity."""
        return self._action

    @value.setter
    def value(self, v: np.ndarray) -> None:
        v = experiment.applybounds(v, self.space.low, self.space.high)  # type: ignore
        self._action = v


class PriceAction(Action):
    """The price of each query type.

    Use "price" as the "kind" of action in the config.
    """

    def __init__(
        self, *, low: float, high: float, shape: tuple[int, ...], seed: int
    ) -> None:
        super().__init__(low=low, high=high, shape=shape, seed=seed)


class PriceMultiplierAction(Action):
    """The price multiplier and base price of each query type.

    Use "pricemultiplier" as the "kind" of action in the config.

    Attributes:
        baseprice (np.ndarray): The base price for each product.
    """

    def __init__(
        self,
        *,
        low: float,
        high: float,
        shape: tuple[int, ...],
        seed: int,
        baseprice: np.ndarray
    ) -> None:
        super().__init__(low=low, high=high, shape=shape, seed=seed)
        self.baseprice = baseprice

    @property
    def value(self) -> np.ndarray:
        return self._action

    @value.setter
    def value(self, v: np.ndarray) -> None:
        v = experiment.applybounds(v, self.space.low, self.space.high)  # type: ignore
        self._action = v


class BudgetAction(Action):
    """The budget across all queries.

    Use "budget" as the "kind" of action in the config.
    """

    def __init__(
        self, *, low: float, high: float, shape: tuple[int, ...], seed: int
    ) -> None:
        super().__init__(low=low, high=high, shape=shape, seed=seed)


def actionfactory(
    *, kind: str, low: float, high: float, shape: tuple[int, ...], seed: int, **kwargs
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
        seed (int): The seed of the random action generator.

    Returns:
        Action: An instantiated action.
    """
    states = {
        "price": PriceAction,
        "budget": BudgetAction,
        "pricemultiplier": PriceMultiplierAction,
    }
    return experiment.factory(
        kind, states, low=low, high=high, shape=shape, seed=seed, **kwargs
    )
