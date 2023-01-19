# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from multipledispatch import dispatch
from numpy._typing import NDArray

from simulation.entity.state import *
from simulation.entity.action import *


@dispatch(PriceState, PriceAction)
def dynamics(s: PriceState, a: PriceAction) -> None:  # type: ignore
    """Update the state given the action.

    In this case, the new state is just the new action.

    Arguments:
        s (PriceState): The previous state
        a (PriceAction): The current action
    """
    s.value = a.value


@dispatch(PriceState, PriceMultiplierAction)
def dynamics(s: PriceState, a: PriceMultiplierAction) -> None:  # type: ignore
    """Update the state given the action.

    In this case, the new state is the price multipliers times the base price.

    Arguments:
        s (PriceState): The previous state
        a (PriceAction): The current action
    """
    s.value = a.value * a.baseprice


@dispatch(BudgetState, BudgetAction)
def dynamics(s: BudgetState, a: BudgetAction) -> None:  # type: ignore
    """Update the state given the action.

    In this case, the new state is just the new action.

    Arguments:
        s (PriceState): The previous state
        a (PriceAction): The current action
    """
    s.value = a.value
