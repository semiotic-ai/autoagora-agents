# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import sacred

from simulation.environment import Environment

simulation_ingredient = sacred.Ingredient("simulation")


@simulation_ingredient.capture
def environment(
    distributor: dict[str, Any],
    entities: list[dict[str, Any]],
    ntimesteps: int,
    nepisodes: int,
    **kwargs
) -> Environment:
    """Construct an environment from the simulation config.

    Arguments:
        distributor (dict[str, Any]): The config for the query distributor.
        entities (list[dict[str, Any]]): The configs for each group of entities.
        nepisodes (int): How many episodes to run.
        ntimesteps (int): How many timesteps to run each episode for.

    Returns:
        Environment: An instantiated simulation environment.
    """
    return Environment(
        distributor=distributor,
        entities=entities,
        ntimesteps=ntimesteps,
        nepisodes=nepisodes,
    )
