# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import numpy as np

from simulation.dynamics import dynamics
from simulation.entity import Agent, Entity, entitygroupfactory
from simulation.isa import isafactory
from simulation.observation import observationfactory
from simulation.reward import rewardfactory


class Environment(gymnasium.Env):
    """The AutoAgora Environment.

    Keyword Arguments:
        isa (dict[str, Any]): The config for the ISA.
        entities (list[dict[str, Any]]): The configs for each group of entities.

    Attributes:
        groups (dict[str, list[Entity]]): A mapping from group names to the entities in
            that group.
        nepisodes (int): How many episodes to run.
        ntimesteps (int): How many timesteps to run each episode for.
        t (int): The current timestep.
        _rewards (dict[str, Reward]): A mapping from group names to the reward function
            of entities in that group.
        _observations (dict[str, Observation]) A mapping from group names to that group's
            observation function.
    """

    def __init__(
        self,
        *,
        isa: dict[str, Any],
        entities: list[dict[str, Any]],
        ntimesteps: int,
        nepisodes: int
    ) -> None:
        super().__init__()
        # Create entities
        self.groups = {e["group"]: entitygroupfactory(**e) for e in entities}
        self.nepisodes = nepisodes
        self.ntimesteps = ntimesteps
        self.t = 0
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

    def reset(
        self,
    ) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, bool]
    ]:
        """Reset the environment.

        Returns:
            observation (dict[str, np.ndarray]): The observations of the agents.
                Each entry in the dictionary maps an agent to its observation.
            reward (dict[str, float]): The rewards of the agents. Each entry in the
                dictionary maps an agent to its reward.
            done (dict[str, bool]): False if an agent is not done. True if it is. Each
                entry in the dictionary maps an agent to its done state.
        """
        self.t = 0
        for group in self.groups.values():
            for entity in group:
                entity.reset()

        return self.observation, self.action, self.reward, self.done

    def step(
        self, *, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], dict[str, bool]
    ]:
        """Step the environment forward given a set of actions.

        Keyword Arguments:
            actions (dict[str, list[np.ndarray]]): The action of each agent.
                The mapping is between group names and lists of actions.

        Returns:
            observation (dict[str, np.ndarray]): The observations of the agents.
                Each entry in the dictionary maps an agent to its observation.
            action (dict[str, np.ndarray]): The actions of the agents.
                Each entry in the dictionary maps an agent to its action.
            reward (dict[str, float]): The rewards of the agents. Each entry in the
                dictionary maps an agent to its reward.
            done (dict[str, bool]): False if an agent is not done. True if it is. Each
                entry in the dictionary maps an agent to its done state.
        """
        self.t += 1
        # Update agent actions
        for agent in self.agentslist:
            agent.action.value = actions[agent.name]

        # Update states
        for agent in self.agentslist:
            dynamics(agent.state, agent.action)  # type: ignore

        self.isa(entities=self.groups)

        return self.observation, self.action, self.reward, self.done

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
    def observation(self) -> dict[str, np.ndarray]:
        """The observations of all agents in the environment."""
        d = {}
        for (group, ags) in self.agents.items():
            obsfn = self._observations[group]
            for a in ags:
                d[a.name] = obsfn(agent=a, entities=self.groups)
        return d

    @property
    def reward(self) -> dict[str, float]:
        """The rewards of all agents in the environment."""
        d = {}
        for (group, ags) in self.agents.items():
            rewfn = self._rewards[group]
            for a in ags:
                d[a.name] = rewfn(agent=a, entities=self.groups)
        return d

    @property
    def done(self) -> dict[str, bool]:
        """Whether each agent is done.

        In our case, agents are only done if the episode is finished.
        """
        d = {}
        for a in self.agentslist:
            d[a.name] = self.isfinished()
        return d

    @property
    def action(self) -> dict[str, np.ndarray]:
        """Each agent's action."""
        d = {}
        for a in self.agentslist:
            d[a.name] = a.action.value
        return d

    def isfinished(self) -> bool:
        """True if t >= ntimesteps. Else false."""
        return self.t >= self.ntimesteps
