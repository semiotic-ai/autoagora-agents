# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from jax import lax
from jax import random as jrand
import jax.numpy as jnp
from jax._src.typing import ArrayLike
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
    def sample(self) -> ArrayLike:
        """ArrayLike: Sample the gaussian distribution."""
        pass

    @abstractmethod
    def logprob(self, x: ArrayLike) -> ArrayLike:
        """The log probability of the PDF at x.

        Arguments:
            x (ArrayLike): A sample.

        Returns:
            ArrayLike: The log probability.
        """
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
        mean (ArrayLike): The clamped mean of the distribution.
        initial_mean (ArrayLike): The means of each gaussian distribution.
        minmean (ArrayLike): The minimum value the mean can take on.
        maxmean (ArrayLike): The maximum value the mean can take on.
        stddev (ArrayLike): The clamped standard deviation of the distribution.
        intial_stddev (ArrayLike): The standard deviations of each gaussian
            distribution.
        minstddev (ArrayLike): The minimum value the standard deviation can take on.
        maxstddev (ArrayLike): The maximum value the standard deviation can take on.
        shape (tuple[int...]): The shape of the vector to return. Normally, it's something
            like (nproducts,)
        seed (int): The random seed of the gaussian.
        key (ArrayLike): A key for seeding sampling.
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
        self._logstddev = jnp.log(self.initial_stddev)

        self.seed = seed
        self.key = jrand.PRNGKey(seed)
        self.shape = self.initial_mean.shape

    @property
    def mean(self) -> ArrayLike:
        return lax.clamp(self.minmean, self._mean, self.maxmean)

    @property
    def stddev(self) -> ArrayLike:
        return lax.clamp(self.minstddev, jnp.exp(self._logstddev), self.maxstddev)

    def reset(self) -> None:
        self._mean = self.initial_mean
        self._logstddev = jnp.log(self.initial_stddev)

    def sample(self) -> ArrayLike:
        v = self.mean + jnp.multiply(
            self.stddev, jrand.normal(self.key, shape=self.shape)
        )
        # Update key
        _, self.key = jrand.split(self.key)
        return v

    def logprob(self, x: ArrayLike) -> ArrayLike:
        return norm.logpdf(x, loc=self.mean, scale=self.stddev)


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
        mean (ArrayLike): The clamped mean of the distribution.
        initial_mean (ArrayLike): The means of each gaussian distribution.
        minmean (ArrayLike): The minimum value the mean can take on.
        maxmean (ArrayLike): The maximum value the mean can take on.
        stddev (ArrayLike): The clamped standard deviation of the distribution.
        intial_stddev (ArrayLike): The standard deviations of each gaussian
            distribution.
        minstddev (ArrayLike): The minimum value the standard deviation can take on.
        maxstddev (ArrayLike): The maximum value the standard deviation can take on.
        scalefactor (ArrayLike): The scale factor for each gaussian distribution.
        invscalefactor (ArrayLike): The inverse scale factor for each gaussian
            distribution.
        shape (tuple[int...]): The shape of the vector to return. Normally, it's something
            like (nproducts,)
        seed (int): The random seed of the gaussian.
        key (ArrayLike): A key for seeding sampling.
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
        self._logstddev = jnp.log(self.initial_stddev)

        self.seed = seed
        self.key = jrand.PRNGKey(seed)
        self.shape = self.initial_mean.shape  # type: ignore

    @property
    def invscalefactor(self) -> ArrayLike:
        return 1 / self.scalefactor

    @property
    def mean(self) -> ArrayLike:
        return lax.clamp(self.minmean, self._mean, self.maxmean)

    @property
    def stddev(self) -> ArrayLike:
        return lax.clamp(self.minstddev, jnp.exp(self._logstddev), self.maxstddev)

    def inversescale(self, x: ArrayLike) -> ArrayLike:
        """Apply the inverse scaling operation to x."""
        return jnp.log(jnp.multiply(self.invscalefactor, x))

    def scale(self, x: ArrayLike) -> ArrayLike:
        """Apply the scaling operation to x."""
        return jnp.multiply(self.scalefactor, jnp.exp(x))

    def reset(self) -> None:
        self._mean = self.initial_mean
        self._logstddev = jnp.log(self.initial_stddev)

    def sample(self) -> ArrayLike:
        """Sample and return values in the scaled space."""
        return self.scale(self.unscaledsample())

    def unscaledsample(self) -> ArrayLike:
        """Sample and return values in the unscaled space."""
        v = self.mean + jnp.multiply(
            self.stddev, jrand.normal(self.key, shape=self.shape)
        )
        # Update key
        _, self.key = jrand.split(self.key)
        return v

    def logprob(self, x: ArrayLike) -> ArrayLike:
        """The log probability of the PDF at x.

        Arguments:
            x (ArrayLike): A sample in the unscaled space.

        Returns:
            ArrayLike: The log probability.
        """
        return norm.logpdf(x, loc=self.mean, scale=self.stddev)


class DegenerateDistribution(Distribution):
    """A degenerate (deterministic) distribution.

    Keyword Arguments:
        initial_value (list[float]): The initial value of the distribution.
        minvalue (list[float]): The minimum value of the distribution.
        maxvalue (list[float]): The maximum value of the distribution.

    Attributes:
        initial_value (ArrayLike): The initial value of the distribution.
        minvalue (ArrayLike): The minimum value of the distribution.
        maxvalue (ArrayLike): The maximum value of the distribution.
        value (ArrayLike): The clamped value of the distribution.
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
    def value(self) -> ArrayLike:
        return lax.clamp(self.minvalue, self._value, self.maxvalue)

    def reset(self) -> None:
        self._value = self.initial_value

    def sample(self) -> ArrayLike:
        return self.value

    def logprob(self, x: ArrayLike) -> ArrayLike:
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
