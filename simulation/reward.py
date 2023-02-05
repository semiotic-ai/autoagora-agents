# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import experiment

from simulation.entity import Agent, Entity


class Reward:
    def __init__(self) -> None:
        pass

    def __call__(self, *, agent: Agent, entities: dict[str, list[Entity]]) -> float:
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

    def __call__(self, *, agent: Agent, entities: dict[str, list[Entity]]) -> float:
        return self._reward(agent=agent, entities=entities) * self.multiplier


class TrafficReward(RewardDecorator):
    """A reward based on how much traffic the agent sends/receives."""

    def __init__(self, *, reward: Reward, multiplier: float) -> None:
        super().__init__(reward=reward, multiplier=multiplier)

    def __call__(self, *, agent: Agent, entities: dict[str, list[Entity]]) -> float:
        return agent.state.fee * self.multiplier + self._reward(  # type: ignore
            agent=agent, entities=entities
        )


class SumRegretRatio(RewardDecorator):
    """A reward based on the fees earned over the total possible fees.

    Attributes:
        fromgroup (str): The group name of the entities paying for queries. Probably
            "consumer" or something similar.
    """

    def __init__(self, *, reward: Reward, multiplier: float, fromgroup: str) -> None:
        super().__init__(reward=reward, multiplier=multiplier)
        self.fromgroup = fromgroup

    def __call__(self, *, agent: Agent, entities: dict[str, list[Entity]]) -> float:
        consumers = entities[self.fromgroup]
        denom = np.sum([np.multiply(c.state.value, c.state.traffic) for c in consumers])
        val = (agent.state.fee / denom) * self.multiplier  # type: ignore
        return val + self._reward(agent=agent, entities=entities)


def rewardfactory(*, rewards: list[dict]) -> Reward:
    """Instantiate a reward object.

    Keyword Arguments:
        rewards (list[dict]): A list of the configs for each reward that make up the
            aggregate reward. Each config must contain the "kind" keyword, wherein
            "kind" can be:
                "traffic"
                "sumregretratio"

    Returns:
        Reward: The reward object
    """
    rdict = {"traffic": TrafficReward, "sumregretratio": SumRegretRatio}
    r = Reward()
    for config in rewards:
        config["reward"] = r
        r = experiment.decoratorfactoryhelper(d=rdict, **config)  # type: ignore

    return r
