# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import sacred

from autoagora_agents.controller import Controller

algorithm_ingredient = sacred.Ingredient("algorithm")


@algorithm_ingredient.capture
def controller(*, agents: list[dict[str, Any]], seed: int, **kwargs) -> Controller:
    """Construct a controller from the algorithm config.

    Keyword Arguments:
        agents (list[dict[str, Any]]): The configs for the agents in each group.
        seed (int): The random seed.

    Returns:
        Controller: An instantiated controller for stepping and updating agents together.
    """
    return Controller(agents=agents, seed=seed)
