# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import numpy as np

from simulation.entity import entitygroupfactory, Agent, Entity
from simulation.reward import rewardfactory
from simulation.observation import observationfactory
from simulation.dynamics import dynamics
from simulation.isa import isafactory


class Environment(gymnasium.Env):
    """The AutoAgora Environment.

    Keyword Arguments:
        isa (dict[str, Any]): The config for the ISA.
        entities (list[dict[str, Any]]): The configs for each group of entities.

    Attributes:
        groups (dict[str, list[Entity]]): A mapping from group names to the entities in
            that group.
        _rewards (dict[str, Reward]): A mapping from group names to the reward function
            of entities in that group.
        _observations (dict[str, Observation]) A mapping from group names to that group's
            observation function.
    """

    def __init__(self, *, isa: dict[str, Any], entities: list[dict[str, Any]]) -> None:
        super().__init__()
        # Create entities
        self.groups = {e["group"]: entitygroupfactory(**e) for e in entities}
        self._rewards = {
            e["group"]: rewardfactory(rewards=e["reward"])
            for e in entities
            if e["kind"] == "agent"
        }
        self._observations = {
            e["group"]: observationfactory(observations=e["observation"])
            for e in entities
            if e["kind"] == "agent"
        }
        self.isa = isafactory(**isa)

    def reset(self) -> dict[str, list[np.ndarray]]:
        """Reset the environment.

        Returns:
            dict[str, list[np.ndarray]]: The initial observations of the agents.
        """
        for group in self.groups.values():
            for entity in group:
                entity.reset()

        return self.observation

    def step(
        self, *, actions: dict[str, list[np.ndarray]]
    ) -> tuple[
        dict[str, list[np.ndarray]], dict[str, list[float]], dict[str, np.ndarray]
    ]:
        """Step the environment forward given a set of actions.

        Keyword Arguments:
            actions (dict[str, list[np.ndarray]]): The action of each agent.
                The mapping is between group names and lists of actions.

        Returns:
            observation (dict[str, list[np.ndarray]]): The observations of the agents.
                Each entry in the dictionary maps a group to a list of observations
                for agents in that group.
            reward (dict[str, float]): The rewards of the agents. Each entry in the
                dictionary maps a group to a list of rewards for agents in that group.
            done (dict[str, np.ndarray]): 0 if an agent is not done. 1 if it is. Each
                entry in the dictionary maps a group to a vector of dones for agents in
                that group.
        """
        # Update agent actions
        for (group, ags) in self.agents.items():
            print(group)
            acts = actions[group]
            for i, ag in enumerate(ags):
                ag.action.value = acts[i]

        # Update states
        for agent in self.agentslist:
            dynamics(agent.state, agent.action)  # type: ignore

        self.isa(entities=self.groups)

        return self.observation, self.reward, self.done

    def render(self):
        """Rendering is not part of the simulation framework."""
        raise NotImplementedError("Rendering is handled by a separate library.")

    def close(self):
        """Closing is not part of the simulation framework."""
        raise NotImplementedError

    @property
    def agents(self) -> dict[str, list[Agent]]:
        """The agents in the environment."""
        return {k: v for (k, v) in self.groups.items() if type(v[0]) == Agent}  # type: ignore

    @property
    def agentslist(self) -> list[Agent]:
        """The agents in the environment as a list."""
        ags = []
        for group in self.agents.values():
            ags.extend(group)
        return ags

    @property
    def entities(self) -> dict[str, list[Entity]]:
        """The entities in the environment."""
        return {k: v for (k, v) in self.groups.items() if type(v[0]) == Entity}  # type: ignore

    @property
    def observation(self) -> dict[str, list[np.ndarray]]:
        """The observations of all agents in the environment."""
        d = {}
        for (group, ags) in self.agents.items():
            obsfn = self._observations[group]
            obs = [obsfn(agent=a, entities=self.groups) for a in ags]
            d[group] = obs

        return d

    @property
    def reward(self) -> dict[str, list[float]]:
        """The rewards of all agents in the environment."""
        d = {}
        for (group, ags) in self.agents.items():
            rewfn = self._rewards[group]
            rew = [rewfn(agent=a, entities=self.groups) for a in ags]
            d[group] = rew

        return d

    @property
    def done(self) -> dict[str, np.ndarray]:
        """Whether each agent is done. In our case, agents are never done, so always 0."""
        return {k: np.zeros(len(v)) for (k, v) in self.agents.items()}
