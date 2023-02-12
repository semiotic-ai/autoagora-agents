# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from simulation.isa import SoftmaxISA

from ..fixture import *


def test_environment_construction(env):
    # Two groups created
    assert len(env.groups) == 2
    assert isinstance(env.isa, SoftmaxISA)


def test_environment_reset(env):
    # Change the state of an indexer
    env.groups["indexer"][0].state.value = np.array([3])
    env.t = 100
    _ = env.reset()
    assert env.groups["indexer"][0].state.value == np.ones(1)
    assert env.t == 0


def test_environment_agents(env):
    agents = tuple(env.agents.keys())
    assert agents == ("indexer",)


def test_environment_entities(env):
    agents = tuple(env.entities.keys())
    assert agents == ("consumer",)


def test_environment_agentslist(env):
    agents = env.agentslist
    assert len(agents) == 2
    assert agents[0].group == "indexer"


def test_environment_observation(env):
    assert env.observation["indexer_0"].size == 0
    assert env.observation["indexer_1"].size == 0


def test_environment_reward(env):
    assert env.reward["indexer_0"] == 0.0
    assert env.reward["indexer_1"] == 0.0
    env.groups["indexer"][0].state.traffic = np.array([2])
    assert env.reward["indexer_0"] == 2.0
    assert env.reward["indexer_1"] == 0.0


def test_environment_isfinished(env):
    assert not env.isfinished()
    env.t = 10
    assert env.isfinished()


def test_environment_done(env):
    assert not env.done["indexer_0"]
    assert not env.done["indexer_1"]
    env.t = 10
    assert env.done["indexer_0"]
    assert env.done["indexer_1"]


def test_environment_render(env):
    with pytest.raises(NotImplementedError):
        env.render()


def test_environment_close(env):
    with pytest.raises(NotImplementedError):
        env.close()


def test_environment_step(env):
    # Change the price of an indexer
    obs, rew, done = env.step(
        actions={"indexer_0": np.array([0.25]), "indexer_1": np.array([3])}
    )
    assert env.groups["indexer"][0].state.value == np.array([0.25])
    assert rew["indexer_0"] == 0.25
    assert rew["indexer_1"] == 0.0
