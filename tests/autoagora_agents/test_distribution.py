# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
from jax import numpy as jnp

from autoagora_agents import distribution


@pytest.fixture
def gaussianconfig():
    return {
        "kind": "gaussian",
        "seed": 0,
        "initial_mean": [1.0],
        "initial_stddev": [0.5],
        "minmean": [0.0],
        "maxmean": [2.0],
        "minstddev": [0.1],
        "maxstddev": [1.0],
    }


@pytest.fixture
def degenerateconfig():
    return {
        "kind": "degenerate",
        "initial_value": [1.0],
        "minvalue": [0.0],
        "maxvalue": [2.0],
    }


@pytest.fixture
def scaledgaussianconfig():
    return {
        "kind": "scaledgaussian",
        "seed": 0,
        "initial_mean": [1.0],
        "initial_stddev": [1.0],
        "minmean": [1.0],
        "maxmean": [5.0],
        "minstddev": [0.1],
        "maxstddev": [1.0],
        "scalefactor": [1.0],
    }


def test_gaussiandistribution_reset(gaussianconfig):
    dist = distribution.distributionfactory(**gaussianconfig)
    v = dist.mean  # type: ignore
    dist._mean = jnp.array([2.0])  # type: ignore
    assert not jnp.allclose(v, dist.mean)  # type: ignore
    dist.reset()
    assert jnp.allclose(v, dist.mean)  # type: ignore


def test_gaussiandistribution_clamping(gaussianconfig):
    dist = distribution.distributionfactory(**gaussianconfig)
    dist._mean = jnp.array([5.0])  # type: ignore
    assert jnp.allclose(dist.mean, jnp.array([2]))  # type: ignore
    dist._logstddev = jnp.array([5.0])  # type: ignore
    assert jnp.allclose(dist.stddev, jnp.array([1]))  # type: ignore


def test_gaussiandistribution_sample(gaussianconfig):
    dist = distribution.distributionfactory(**gaussianconfig)
    samples = jnp.array([dist.sample() for _ in range(1000)])
    assert jnp.allclose(jnp.std(samples), jnp.array(0.5), atol=1e-2)
    assert jnp.allclose(jnp.mean(samples), jnp.array(1.0), atol=1e-2)


def test_degeneratedistribution_reset(degenerateconfig):
    dist = distribution.distributionfactory(**degenerateconfig)
    v = dist.value  # type: ignore
    dist._value = jnp.array([2.0])  # type: ignore
    dist.reset()
    assert jnp.allclose(v, dist.value)  # type: ignore


def test_degeneratedistribution_clamping(degenerateconfig):
    dist = distribution.distributionfactory(**degenerateconfig)
    dist._value = jnp.array([5.0])  # type: ignore
    assert jnp.allclose(dist.value, jnp.array([2]))  # type: ignore


def test_degeneratedistribution_sample(degenerateconfig):
    dist = distribution.distributionfactory(**degenerateconfig)
    samples = jnp.array([dist.sample() for _ in range(10)])
    assert jnp.sum(samples) == 10


def test_scaledgaussiandistribution_reset(scaledgaussianconfig):
    dist = distribution.distributionfactory(**scaledgaussianconfig)
    v = dist.mean  # type: ignore
    dist._mean = jnp.array([2.0])  # type: ignore
    assert not jnp.allclose(v, dist.mean)  # type: ignore
    dist.reset()
    assert jnp.allclose(v, dist.mean)  # type: ignore


def test_scaledgaussiandistribution_clamping(scaledgaussianconfig):
    dist = distribution.distributionfactory(**scaledgaussianconfig)
    dist._mean = jnp.array([-1.0])  # type: ignore
    assert jnp.allclose(dist.mean, jnp.array([0.0]))  # type: ignore
    dist._logstddev = jnp.array([-100.0])  # type: ignore
    assert jnp.allclose(dist.stddev, jnp.array([0.1]))  # type: ignore


def test_scaledgaussiandistribution_unscaledsample(scaledgaussianconfig):
    dist = distribution.distributionfactory(**scaledgaussianconfig)
    samples = jnp.array([dist.unscaledsample() for _ in range(1000)])  # type: ignore
    assert jnp.allclose(jnp.std(samples), jnp.array(1.0), atol=1e-1)
    assert jnp.allclose(jnp.mean(samples), jnp.array(0.0), atol=1e-1)


def test_scaledgaussiandistribution_scale(scaledgaussianconfig):
    dist = distribution.distributionfactory(**scaledgaussianconfig)
    jnp.allclose(dist.scale(jnp.array([0.0])), jnp.array([1.0]))  # type: ignore
