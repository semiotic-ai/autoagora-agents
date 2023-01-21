# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from statistics import mean
from typing import Union

import numpy
import pytest
import torch

from autoagora_agents.agent_factory import AgentFactory
from autoagora_agents.policy_mixins import Policy


class TestScaledGaussianAgent:
    @pytest.mark.unit
    @pytest.mark.parametrize("test_input", [0.3, torch.tensor(0.3)])
    def test_scaled_gaussian_scaling(self, test_input: Union[float, torch.Tensor]):
        """Test the scale - inv_scale operation."""
        agent = AgentFactory(
            agent_name="test_agent", agent_section={"action": "scaled_gaussian"}
        )
        assert numpy.isclose(
            test_input,
            agent.inverse_bid_scale(agent.bid_scale(test_input)),
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "policy_name",
        [
            "vpg",
            "ppo",
            "rolling_ppo",
        ],
    )
    @pytest.mark.parametrize(
        "initial_mean, initial_stddev, gauss_min, gauss_max",
        [
            [1e-6, 1e-1, 0.5e-6, 1.5e-6],
        ],
    )
    def test_zero_mean_bead(
        self,
        policy_name: Policy,
        initial_mean: float,
        initial_stddev: float,
        gauss_min: float,
        gauss_max: float,
    ):
        """Tests bid"""
        # Create agent.
        agent = AgentFactory(
            agent_name="test_agent",
            agent_section={
                "action": {
                    "type": "scaled_gaussian",
                    "initial_mean": initial_mean,
                    "initial_stddev": initial_stddev,
                },
                "policy": policy_name,
            },
        )

        # Get number of bids and average.
        scaled_bids = 0.0
        for _ in range(1000):
            scaled_bids += agent.get_action()
        mean_scaled_bids = scaled_bids / 1000
        assert (mean_scaled_bids >= gauss_min) and (mean_scaled_bids <= gauss_max)

    def test_save_reload_mean_stddev(self):
        initial_mean = 1e-6
        initial_stddev = 1e-7

        agent = AgentFactory(
            agent_name="test_agent",
            agent_section={
                "action": {
                    "type": "scaled_gaussian",
                    "initial_mean": initial_mean,
                    "initial_stddev": initial_stddev,
                },
                "policy": "rolling_ppo",
            },
        )

        save_mean = agent.bid_scale(agent.mean().item())
        save_stddev = agent.stddev().item()

        numpy.testing.assert_approx_equal(
            initial_mean,
            save_mean,
            err_msg="Initial mean and saved mean are not equal.",
        )
        numpy.testing.assert_approx_equal(
            initial_stddev,
            save_stddev,
            err_msg="Initial stddev and saved stddev are not equal.",
        )
