# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import lax
from jax import random as jrand
from jax.scipy.stats import norm

import experiment


class Distribution(ABC):
    """The base class for distributions."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self) -> None:
        """Reset the distribution to its initial values."""
        pass

    @abstractmethod
    def sample(self) -> jnp.ndarray:
        """jnp.ndarray: Sample the gaussian distribution."""
        pass

    @abstractmethod
    def logprob(self, x: jnp.ndarray) -> jnp.ndarray:
        """The log probability of the PDF at x.

        Arguments:
            x (jnp.ndarray): A sample.

        Returns:
            jnp.ndarray: The log probability.
        """
        pass

    @abstractmethod
    def entropy(self) -> jnp.ndarray:
        """The entropy of the distribution."""
        pass


class GaussianDistribution(Distribution):
    """A Gaussian distribution.

    Keyword Arguments:
        seed (int): The random seed of the gaussian.
        intial_mean (list[float]): The means of each gaussian distribution. For example,
            for multi-product, you would set one initial mean per product.
        minmean (list[float]): The minimum value the mean can take on.
        maxmean (list[float]): The maximum value the mean can take on.
        intial_stddev (list[float]): The standard deviations of each gaussian
            distribution.
        minstddev (list[float]): The minimum value the standard deviation can take on.
        maxstddev (list[float]): The maximum value the standard deviation can take on.

    Attributes:
        mean (jnp.ndarray): The clamped mean of the distribution.
        initial_mean (jnp.ndarray): The means of each gaussian distribution.
        minmean (jnp.ndarray): The minimum value the mean can take on.
        maxmean (jnp.ndarray): The maximum value the mean can take on.
        stddev (jnp.ndarray): The clamped standard deviation of the distribution.
        intial_stddev (jnp.ndarray): The standard deviations of each gaussian
            distribution.
        minstddev (jnp.ndarray): The minimum value the standard deviation can take on.
        maxstddev (jnp.ndarray): The maximum value the standard deviation can take on.
        shape (tuple[int...]): The shape of the vector to return. Normally, it's something
            like (nproducts,)
        seed (int): The random seed of the gaussian.
        key (jnp.ndarray): A key for seeding sampling.
    """

    def __init__(
        self,
        *,
        seed: int,
        initial_mean: list[float],
        minmean: list[float],
        maxmean: list[float],
        initial_stddev: list[float],
        minstddev: list[float],
        maxstddev: list[float],
    ) -> None:
        super().__init__()

        self.initial_mean = jnp.array(initial_mean)
        self.maxmean = jnp.array(maxmean)
        self.minmean = jnp.array(minmean)
        self._mean = self.initial_mean

        self.initial_stddev = jnp.array(initial_stddev)
        self.maxstddev = jnp.array(maxstddev)
        self.minstddev = jnp.array(minstddev)
        self.logstddev = jnp.log(self.initial_stddev)

        self.seed = seed
        self.key = jrand.PRNGKey(seed)
        self.shape = self.initial_mean.shape

    @property
    def mean(self) -> jnp.ndarray:
        return lax.clamp(self.minmean, self._mean, self.maxmean)

    @property
    def stddev(self) -> jnp.ndarray:
        return lax.clamp(self.minstddev, jnp.exp(self.logstddev), self.maxstddev)

    def reset(self) -> None:
        self._mean = self.initial_mean
        self.logstddev = jnp.log(self.initial_stddev)

    def sample(self) -> jnp.ndarray:
        v = self.mean + jnp.multiply(
            self.stddev, jrand.normal(self.key, shape=self.shape)
        )
        # Update key
        _, self.key = jrand.split(self.key)
        return v

    def logprob(self, x: jnp.ndarray) -> jnp.ndarray:
        return norm.logpdf(x, loc=self.mean, scale=self.stddev)

    def entropy(self) -> jnp.ndarray:
        return 0.5 + 0.5 * jnp.log(2 * math.pi) + jnp.log(self.mean)


class ScaledGaussianDistribution(Distribution):
    """A Gaussian distribution wherein the gaussian is in a scaled space.

    In the scaled space, the mean is multiplied by the inverse scale factor and then put
    into log space. This also applies to the bounds on the mean below.

    Keyword Arguments:
        seed (int): The random seed of the gaussian.
        intial_mean (list[float]): The means of each gaussian distribution. For example,
            for multi-product, you would set one initial mean per product.
        minmean (list[float]): The minimum value the mean can take on.
        maxmean (list[float]): The maximum value the mean can take on.
        intial_stddev (list[float]): The standard deviations of each gaussian
            distribution.
        minstddev (list[float]): The minimum value the standard deviation can take on.
        maxstddev (list[float]): The maximum value the standard deviation can take on.
        scalefactor (list[float]): The scale factor for each gaussian distribution.

    Attributes:
        mean (jnp.ndarray): The clamped mean of the distribution.
        initial_mean (jnp.ndarray): The means of each gaussian distribution.
        minmean (jnp.ndarray): The minimum value the mean can take on.
        maxmean (jnp.ndarray): The maximum value the mean can take on.
        stddev (jnp.ndarray): The clamped standard deviation of the distribution.
        intial_stddev (jnp.ndarray): The standard deviations of each gaussian
            distribution.
        minstddev (jnp.ndarray): The minimum value the standard deviation can take on.
        maxstddev (jnp.ndarray): The maximum value the standard deviation can take on.
        scalefactor (jnp.ndarray): The scale factor for each gaussian distribution.
        invscalefactor (jnp.ndarray): The inverse scale factor for each gaussian
            distribution.
        shape (tuple[int...]): The shape of the vector to return. Normally, it's something
            like (nproducts,)
        seed (int): The random seed of the gaussian.
        key (jnp.ndarray): A key for seeding sampling.
    """

    def __init__(
        self,
        *,
        seed: int,
        initial_mean: list[float],
        minmean: list[float],
        maxmean: list[float],
        initial_stddev: list[float],
        minstddev: list[float],
        maxstddev: list[float],
        scalefactor: list[float],
    ) -> None:
        super().__init__()
        self.scalefactor = jnp.array(scalefactor)

        self.initial_mean = self.inversescale(jnp.array(initial_mean))
        self.maxmean = self.inversescale(jnp.array(maxmean))
        self.minmean = self.inversescale(jnp.array(minmean))
        self._mean = self.initial_mean

        self.initial_stddev = jnp.array(initial_stddev)
        self.maxstddev = jnp.array(maxstddev)
        self.minstddev = jnp.array(minstddev)
        self.logstddev = jnp.log(self.initial_stddev)

        self.seed = seed
        self.key = jrand.PRNGKey(seed)
        self.shape = self.initial_mean.shape  # type: ignore

    @property
    def invscalefactor(self) -> jnp.ndarray:
        return 1 / self.scalefactor

    @property
    def mean(self) -> jnp.ndarray:
        return lax.clamp(self.minmean, self._mean, self.maxmean)

    @property
    def stddev(self) -> jnp.ndarray:
        return lax.clamp(self.minstddev, jnp.exp(self.logstddev), self.maxstddev)

    def inversescale(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the inverse scaling operation to x."""
        return jnp.log(jnp.multiply(self.invscalefactor, x))

    def scale(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the scaling operation to x."""
        return jnp.multiply(self.scalefactor, jnp.exp(x))

    def reset(self) -> None:
        self._mean = self.initial_mean
        self.logstddev = jnp.log(self.initial_stddev)

    def sample(self) -> jnp.ndarray:
        """Sample and return values in the scaled space."""
        return self.scale(self.unscaledsample())

    def unscaledsample(self) -> jnp.ndarray:
        """Sample and return values in the unscaled space."""
        v = self.mean + jnp.multiply(
            self.stddev, jrand.normal(self.key, shape=self.shape)
        )
        # Update key
        _, self.key = jrand.split(self.key)
        return v

    def logprob(self, x: jnp.ndarray) -> jnp.ndarray:
        """The log probability of the PDF at x.

        Arguments:
            x (jnp.ndarray): A sample in the scaled space.

        Returns:
            jnp.ndarray: The log probability.
        """
        y = self.inversescale(x)
        return norm.logpdf(y, loc=self.mean, scale=self.stddev)

    def entropy(self) -> jnp.ndarray:
        return 0.5 + 0.5 * jnp.log(2 * math.pi) + jnp.log(self.mean)


class DegenerateDistribution(Distribution):
    """A degenerate (deterministic) distribution.

    Keyword Arguments:
        initial_value (list[float]): The initial value of the distribution.
        minvalue (list[float]): The minimum value of the distribution.
        maxvalue (list[float]): The maximum value of the distribution.

    Attributes:
        initial_value (jnp.ndarray): The initial value of the distribution.
        minvalue (jnp.ndarray): The minimum value of the distribution.
        maxvalue (jnp.ndarray): The maximum value of the distribution.
        value (jnp.ndarray): The clamped value of the distribution.
    """

    def __init__(
        self,
        *,
        initial_value: list[float],
        minvalue: list[float],
        maxvalue: list[float],
    ) -> None:
        super().__init__()
        self.initial_value = jnp.array(initial_value)
        self.minvalue = jnp.array(minvalue)
        self.maxvalue = jnp.array(maxvalue)
        self._value = self.initial_value

    @property
    def value(self) -> jnp.ndarray:
        return lax.clamp(self.minvalue, self._value, self.maxvalue)

    def reset(self) -> None:
        self._value = self.initial_value

    def sample(self) -> jnp.ndarray:
        return self.value

    def logprob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(self._value)

    def entropy(self) -> jnp.ndarray:
        return jnp.zeros_like(self._value)


def distributionfactory(*, kind: str, **kwargs) -> Distribution:
    """Instantiate a new distribution.

    Keyword Arguments:
        kind (str): The type of distribution to instantiate.
            "gaussian" -> GaussianDistribution
            "scaledgaussian" -> ScaledGaussianDistribution
            "degenerate" -> DegenerateDistribution

    Returns:
        Distribution: An instantiated distribution.
    """
    dists = {
        "gaussian": GaussianDistribution,
        "scaledgaussian": ScaledGaussianDistribution,
        "degenerate": DegenerateDistribution,
    }
    return experiment.factory(kind, dists, **kwargs)
