# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import experiment

from simulation.entity import Agent, Entity


class Reward:
    def __init__(self) -> None:
        pass

    def __call__(self, *, agent: Agent, entities: dict[str, Entity]) -> float:
        return 0.0


class RewardDecorator(Reward):
    """The base class for reward decorators.

    Attributes:
        reward (Reward): An instance of :class:`Reward`
        multiplier (float): The reward is scaled by this value.
            Make this negative to make the reward into a penalty.
    """

    def __init__(self, *, reward: Reward, multiplier: float) -> None:
        super().__init__()
        self._reward = reward
        self.multiplier = multiplier

    @property
    def reward(self) -> Reward:
        return self._reward

    def __call__(self, *, agent: Agent, entities: dict[str, Entity]) -> float:
        return self._reward(agent=agent, entities=entities) * self.multiplier


class TrafficReward(RewardDecorator):
    """A reward based on how much traffic the agent sends/receives."""

    def __init__(self, *, reward: Reward, multiplier: float) -> None:
        super().__init__(reward=reward, multiplier=multiplier)

    def __call__(self, *, agent: Agent, entities: dict[str, Entity]) -> float:
        return sum(
            np.multiply(agent.state.value, agent.state.traffic)
        ) * self.multiplier + self._reward(agent=agent, entities=entities)


def rewardfactory(*, rewards: list[dict]) -> Reward:
    """Instantiate a reward object.

    Keyword Arguments:
        rewards (list[dict]): A list of the configs for each reward that make up the
            aggregate reward. Each config must contain the "kind" keyword, wherein
            "kind" can be:
                "traffic"

    Returns:
        Reward: The reward object
    """
    rdict = {"traffic": TrafficReward}
    r = Reward()
    for config in rewards:
        config["reward"] = r
        r = rewardfactoryhelper(d=rdict, **config)  # type: ignore

    return r


def rewardfactoryhelper(*, kind: str, d: dict[str, Reward], **kwargs) -> Reward:
    """Extract "kind" from the config.

    Keyword Arguments:
        kind (str): The kind of reward.
        d (dict[str, Reward]): Maps strings to reward classes

    Returns:
        An instantied reward object.
    """
    return experiment.factory(kind, d, **kwargs)
