# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from autoagora_agents import distribution

from ..fixture import *


def test_gaussiandistribution_reset(gaussianconfig):
    dist = distribution.distributionfactory(**gaussianconfig)
    v = dist.mean  # type: ignore
    dist._mean = torch.tensor([2.0])  # type: ignore
    assert not torch.allclose(v, dist.mean)  # type: ignore
    dist.reset()
    assert torch.allclose(v, dist.mean)  # type: ignore


def test_gaussiandistribution_clamping(gaussianconfig):
    dist = distribution.distributionfactory(**gaussianconfig)
    dist._mean = torch.tensor([5.0])  # type: ignore
    assert torch.allclose(dist.mean, torch.tensor([2.0]))  # type: ignore
    dist._logstddev = torch.tensor([5.0])  # type: ignore
    assert torch.allclose(dist.stddev, torch.tensor([1.0]))  # type: ignore


def test_gaussiandistribution_sample(gaussianconfig):
    dist = distribution.distributionfactory(**gaussianconfig)
    samples = torch.tensor([dist.sample() for _ in range(1000)])
    assert torch.allclose(torch.std(samples), torch.tensor(0.5), atol=1e-1)
    assert torch.allclose(torch.mean(samples), torch.tensor(1.0), atol=1e-1)


def test_degeneratedistribution_reset(degenerateconfig):
    dist = distribution.distributionfactory(**degenerateconfig)
    v = dist.mean  # type: ignore
    dist._mean = torch.tensor([2.0])  # type: ignore
    dist.reset()
    assert torch.allclose(v, dist.mean)  # type: ignore


def test_degeneratedistribution_clamping(degenerateconfig):
    dist = distribution.distributionfactory(**degenerateconfig)
    dist._value = torch.tensor([5.0])  # type: ignore
    assert torch.allclose(dist.mean, torch.tensor([2.0]))  # type: ignore


def test_degeneratedistribution_sample(degenerateconfig):
    dist = distribution.distributionfactory(**degenerateconfig)
    samples = torch.tensor([dist.sample() for _ in range(10)])
    assert torch.sum(samples) == 10


def test_degeneratedistribution_entropy(degenerateconfig):
    dist = distribution.distributionfactory(**degenerateconfig)
    assert torch.sum(dist.entropy()) == 0


def test_scaledgaussiandistribution_reset(scaledgaussianconfig):
    dist = distribution.distributionfactory(**scaledgaussianconfig)
    v = dist.mean  # type: ignore
    dist._mean = torch.tensor([2.0])  # type: ignore
    assert not torch.allclose(v, dist.mean)  # type: ignore
    dist.reset()
    assert torch.allclose(v, dist.mean)  # type: ignore


def test_scaledgaussiandistribution_clamping(scaledgaussianconfig):
    dist = distribution.distributionfactory(**scaledgaussianconfig)
    dist._mean = torch.tensor([-1.0])  # type: ignore
    assert torch.allclose(dist.mean, torch.tensor([0.0]))  # type: ignore
    dist._logstddev = torch.tensor([-100.0])  # type: ignore
    assert torch.allclose(dist.stddev, torch.tensor([0.1]))  # type: ignore


def test_scaledgaussiandistribution_unscaledsample(scaledgaussianconfig):
    dist = distribution.distributionfactory(**scaledgaussianconfig)
    samples = torch.tensor([dist.unscaledsample() for _ in range(1000)])  # type: ignore
    assert torch.allclose(torch.std(samples), torch.tensor(1.0), atol=1e-1)
    assert torch.allclose(torch.mean(samples), torch.tensor(0.0), atol=1e-1)


def test_scaledgaussiandistribution_scale(scaledgaussianconfig):
    dist = distribution.distributionfactory(**scaledgaussianconfig)
    torch.allclose(dist.scale(torch.tensor([0.0])), torch.tensor([1.0]))  # type: ignore
