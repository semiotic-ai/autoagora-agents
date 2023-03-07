# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import random
from typing import Any

import numpy as np
import torch

from autoagora_agents.algorithm import Algorithm, algorithmgroupfactory


class Controller:
    """Holds all algorithms and routes information to each.

    Keyword Arguments:
        agents (list[dict[str, Any]]): A list of the configs for each agent
            group.
        seed (int): The seed for torch.

    Attributes:
        groups (dict[str, Algorithm]): A dictionary mapping agent groups to algorithms.
    """

    def __init__(self, *, agents: list[dict[str, Any]], seed: int) -> None:
        self.groups = {a["group"]: algorithmgroupfactory(**a) for a in agents}
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __call__(
        self,
        *,
        observations: dict[str, np.ndarray],
        actions: dict[str, np.ndarray],
        rewards: dict[str, float],
        dones: dict[str, bool]
    ) -> dict[str, np.ndarray]:
        """Call each algorithm.

        Keyword Arguments:
            observations (dict[str, np.ndarray]): The observations of each agent.
            actions (dict[str, np.ndarray]): The action of each agent.
            rewards (dict[str, float]): The reward received by each agent.
            dones (dict[str, bool]): Whether each agent is done.

        Returns:
            dict[str, np.ndarray]: The next actions of each agent.
        """
        acts = {}
        for alg in self.algorithmslist:
            acts[alg.name] = alg(
                observation=observations[alg.name],
                action=actions[alg.name],
                reward=rewards[alg.name],
                done=dones[alg.name],
            )
        return acts

    def update(self) -> None:
        """Update each algorithm."""
        for alg in self.algorithmslist:
            alg.update()

    @property
    def algorithmslist(self) -> list[Algorithm]:
        """The algorithms for agents in each group."""
        algs = []
        for a in self.groups.values():
            algs.extend(a)
        return algs
