# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from simulation import environment
from simulation.isa import SoftmaxISA

from ..fixture import *


def test_environment_construction(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    # Two groups created
    assert len(env.groups) == 2
    assert isinstance(env.isa, SoftmaxISA)


def test_environment_reset(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    # Change the state of an indexer
    env.groups["indexer"][0].state.value = np.array([3])
    _ = env.reset()
    assert env.groups["indexer"][0].state.value == np.ones(1)


def test_environment_agents(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    agents = tuple(env.agents.keys())
    assert agents == ("indexer",)


def test_environment_entities(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    agents = tuple(env.entities.keys())
    assert agents == ("consumer",)


def test_environment_agentslist(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    agents = env.agentslist
    assert len(agents) == 2
    assert agents[0].group == "indexer"


def test_environment_observation(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    assert np.sum(env.observation["indexer"]) == 0.0


def test_environment_reward(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    assert np.sum(env.reward["indexer"]) == 0.0
    env.groups["indexer"][0].state.traffic = np.array([2])
    assert np.sum(env.reward["indexer"]) == 2.0


def test_environment_done(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    assert np.sum(env.done["indexer"]) == 0.0


def test_environment_render(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    with pytest.raises(NotImplementedError):
        env.render()


def test_environment_close(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    with pytest.raises(NotImplementedError):
        env.close()


def test_environment_step(simulationconfig):
    env = environment(simulationconfig["isa"], simulationconfig["entities"])
    # Change the price of an indexer
    obs, rew, done = env.step(actions={"indexer": [np.array([0.25]), np.array([3])]})
    assert env.groups["indexer"][0].state.value == np.array([0.25])
    assert rew["indexer"][0] == 0.25
    assert rew["indexer"][1] == 0.0
