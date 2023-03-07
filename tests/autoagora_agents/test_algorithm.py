# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from autoagora_agents import algorithm

from ..fixture import *


def test_predetermined(predeterminedconfig):
    agent = algorithm.algorithmgroupfactory(**predeterminedconfig)[0]
    obs = np.zeros(1)
    rew = 1
    act = np.zeros(1)
    for i in range(10):
        act = agent(observation=obs, action=act, reward=rew, done=False)
        if i < 3:
            assert np.array_equiv(np.zeros(1), act)
        elif i >= 6:
            assert np.array_equiv(2 * np.ones(1), act)
        else:
            assert np.array_equiv(np.ones(1), act)

    assert agent.niterations == 10
    agent.reset()
    assert agent.niterations == 0


def test_predetermined_nonzero_first_timestamp(predeterminedconfig):
    predeterminedconfig["timestamps"] = [5, 10, 15]
    with pytest.raises(ValueError):
        _ = algorithm.algorithmgroupfactory(**predeterminedconfig)[0]


def test_predetermined_different_length_lists(predeterminedconfig):
    predeterminedconfig["timestamps"] = [0, 10]
    with pytest.raises(ValueError):
        _ = algorithm.algorithmgroupfactory(**predeterminedconfig)[0]


def test_advantage_reward_std_nan(predeterminedconfig):
    # The config here doesn't matter. We just need to set up some agent to get access to the advantage static method
    agent = algorithm.algorithmgroupfactory(**predeterminedconfig)[0]
    rewards = torch.as_tensor([1.0])
    adv = agent.advantage(rewards)
    assert adv == rewards.unsqueeze(dim=1)


def test_advantage_reward_std_zero(predeterminedconfig):
    # The config here doesn't matter. We just need to set up some agent to get access to the advantage static method
    agent = algorithm.algorithmgroupfactory(**predeterminedconfig)[0]
    rewards = torch.as_tensor([1.0, 1.0])
    adv = agent.advantage(rewards)
    assert all(adv == rewards.unsqueeze(dim=1))


def test_advantage_reward_std_nonzero(predeterminedconfig):
    # The config here doesn't matter. We just need to set up some agent to get access to the advantage static method
    agent = algorithm.algorithmgroupfactory(**predeterminedconfig)[0]
    for _ in range(100):
        rewards = torch.randint(-100, 100, (10,), dtype=torch.float32)
        adv = agent.advantage(rewards)
        # Our definintion of advantage here is essentially just standardising a gaussian
        assert torch.allclose(adv.mean(), torch.zeros(1), atol=1e-2)
        assert torch.allclose(adv.std(), torch.ones(1), atol=1e-2)


def test_bandit_call(vpgbanditconfig):
    agent = algorithm.algorithmgroupfactory(**vpgbanditconfig)[0]
    obs = np.zeros(1)
    act = np.zeros(1)
    rew = 1
    done = False
    act = agent(observation=obs, action=act, reward=rew, done=done)
    assert len(agent.buffer) == 1  # type: ignore
    act = agent(observation=obs, action=act, reward=rew, done=done)
    assert len(agent.buffer) == 2  # type: ignore
    # Buffer is a deque, so shouldn't fill more
    act = agent(observation=obs, action=act, reward=rew, done=done)
    assert len(agent.buffer) == 2  # type: ignore
