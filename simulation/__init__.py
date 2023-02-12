# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import sacred

from simulation.environment import Environment

simulation_ingredient = sacred.Ingredient("simulation")


@simulation_ingredient.capture
def environment(
    isa: dict[str, Any], entities: list[dict[str, Any]], **kwargs
) -> Environment:
    """Construct an environment from the simulation config.

    Arguments:
        isa (dict[str, Any]): The config for the ISA.
        entities (list[dict[str, Any]]): The configs for each group of entities.

    Returns:
        Environment: An instantiated simulation environment.
    """
    return Environment(isa=isa, entities=entities)
