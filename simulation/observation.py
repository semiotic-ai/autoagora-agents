# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import experiment
from simulation.entity import Agent, Entity


class Observation:
    def __init__(self) -> None:
        pass

    def __call__(
        self, *, agent: Agent, entities: dict[str, list[Entity]]
    ) -> np.ndarray:
        return np.array([])


class ObservationDecorator(Observation):
    """The base class for observation decorators.

    Attributes:
        observation (Observation): An instance of :class:`Observation`
    """

    def __init__(self, *, observation: Observation) -> None:
        super().__init__()
        self._observation = observation

    @property
    def observation(self) -> Observation:
        return self._observation

    def __call__(
        self, *, agent: Agent, entities: dict[str, list[Entity]]
    ) -> np.ndarray:
        return self._observation(agent=agent, entities=entities)


class BanditObservation(ObservationDecorator):
    """An empty observation for bandits."""

    def __init__(self, *, observation: Observation) -> None:
        super().__init__(observation=observation)

    def __call__(
        self, *, agent: Agent, entities: dict[str, list[Entity]]
    ) -> np.ndarray:
        return np.append(
            np.array([]), self._observation(agent=agent, entities=entities)
        )


def observationfactory(*, observations: list[dict]) -> Observation:
    """Instantiate an observation object.

    Keyword Arguments:
        observations (list[dict]): A list of the configs for each observation component
            that makes up the full observation. Each config must contain the "kind"
            keyword, wherein "kind" can be:
                "bandit"

    Returns:
        Observation: The observation object
    """
    rdict = {"bandit": BanditObservation}
    o = Observation()
    for config in observations:
        config["observation"] = o
        o = experiment.decoratorfactoryhelper(d=rdict, **config)  # type: ignore

    return o
