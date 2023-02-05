# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import experiment
from simulation.entity import statefactory, actionfactory


class Entity:
    """An entity is an object with a state space, but without an action space.

    Attributes:
        group (str): To which group this entity belongs. E.g., "consumer" or "indexer".
        i (int): The index of the entity.
        name (str): kind_i
        state (State): The state of the entity
        state_space (spaces.Space): The state space of the entity.
    """

    def __init__(self, *, group: str, i: int, state: dict, **kwargs) -> None:
        self.group = group
        self.i = i
        self.name = f"{group}_{i}"
        self.state = statefactory(**state)

    def reset(self) -> None:
        """Reset the entity."""
        self.state.reset()


class Agent(Entity):
    """An entity is an object with a state space and an action space.

    Attributes:
        action (Action): The action taken by the agent
    """

    def __init__(
        self, *, group: str, i: int, state: dict, action: dict, **kwargs
    ) -> None:
        super().__init__(group=group, i=i, state=state)
        self.action = actionfactory(**action)

    def reset(self) -> None:
        """Reset the agent."""
        self.state.reset()


def entitygroupfactory(*, kind: str, count: int, **kwargs) -> list[Entity]:
    """Instantiate new entities of a particular group.

    Keyword Arguments:
        kind (str): "entity" or "agent".
        count (int): The number of entities in this group.

    Returns:
        list[Entity]: A list of instantiated entities.
    """
    edict = {"entity": Entity, "agent": Agent}
    group = [experiment.factory(kind, edict, i=i, **kwargs) for i in range(count)]
    return group
