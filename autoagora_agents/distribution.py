# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod, abstractproperty
from typing import Union

import numpy as np
import torch
from torch import nn

import experiment

ArrayLike = Union[np.ndarray, torch.Tensor]


class Distribution(ABC):
    """The base class for distributions."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self) -> None:
        """Reset the distribution to its initial values."""
        pass

    @abstractproperty
    def initial_mean(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: Initial mean of the distribution."""
        pass

    @abstractproperty
    def initial_stddev(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: Initial standard deviation of the distribution."""
        pass

    @abstractproperty
    def logstddev(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: The log standard deviation of the distribution."""
        pass

    @abstractproperty
    def mean(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: Mean of the distribution."""
        pass

    @abstractproperty
    def unclampedmean(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: Unclamped mean of the distribution."""
        pass

    @abstractproperty
    def stddev(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: Standard deviation of the distribution."""
        pass

    @abstractproperty
    def distribution(self) -> torch.distributions.Distribution:  # type: ignore
        """torch.distributions.Distribution: The torch distribution."""
        pass

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """torch.Tensor: Sample the gaussian distribution."""
        pass

    @abstractmethod
    def logprob(self, x: torch.Tensor) -> torch.Tensor:
        """The log probability of the PDF at x.

        Arguments:
            x (torch.Tensor): A sample.

        Returns:
            torch.Tensor: The log probability.
        """
        pass

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """The entropy of the distribution."""
        pass


class GaussianDistribution(Distribution):
    """A Gaussian distribution.

    Keyword Arguments:
        intial_mean (ArrayLike): The means of each gaussian distribution. For example,
            for multi-product, you would set one initial mean per product.
        minmean (ArrayLike): The minimum value the mean can take on.
        maxmean (ArrayLike): The maximum value the mean can take on.
        intial_stddev (ArrayLike): The standard deviations of each gaussian
            distribution.
        minstddev (ArrayLike): The minimum value the standard deviation can take on.
        maxstddev (ArrayLike): The maximum value the standard deviation can take on.

    Attributes:
        mean (torch.Tensor): The clamped mean of the distribution.
        minmean (torch.Tensor): The minimum value the mean can take on.
        maxmean (torch.Tensor): The maximum value the mean can take on.
        stddev (torch.Tensor): The clamped standard deviation of the distribution.
        minstddev (torch.Tensor): The minimum value the standard deviation can take on.
        maxstddev (torch.Tensor): The maximum value the standard deviation can take on.
    """

    def __init__(
        self,
        *,
        initial_mean: ArrayLike,
        minmean: ArrayLike,
        maxmean: ArrayLike,
        initial_stddev: ArrayLike,
        minstddev: ArrayLike,
        maxstddev: ArrayLike,
    ) -> None:
        super().__init__()

        self._initial_mean = torch.as_tensor(initial_mean)
        self.maxmean = torch.as_tensor(maxmean)
        self.minmean = torch.as_tensor(minmean)
        self._mean = nn.parameter.Parameter(self.initial_mean)

        self._initial_stddev = torch.as_tensor(initial_stddev)
        self.maxstddev = torch.as_tensor(maxstddev)
        self.minstddev = torch.as_tensor(minstddev)
        self._logstddev = nn.parameter.Parameter(torch.log(self.initial_stddev))

    @property
    def mean(self) -> torch.Tensor:
        return torch.clip(self._mean, min=self.minmean, max=self.maxmean)

    @property
    def stddev(self) -> torch.Tensor:
        return torch.clip(
            torch.exp(self.logstddev), min=self.minstddev, max=self.maxstddev
        )

    @property
    def logstddev(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: The log standard deviation of the distribution."""
        return self._logstddev

    @property
    def unclampedmean(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: Unclamped mean of the distribution."""
        return self._mean

    @property
    def initial_mean(self) -> torch.Tensor:
        return self._initial_mean

    @property
    def initial_stddev(self) -> torch.Tensor:
        return self._initial_stddev

    def reset(self) -> None:
        self._mean = nn.parameter.Parameter(self.initial_mean)
        self._logstddev = nn.parameter.Parameter(torch.log(self.initial_stddev))

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Normal(loc=self.mean, scale=self.stddev)

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample().detach().item()

    def logprob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()


class ScaledGaussianDistribution(GaussianDistribution):
    """A Gaussian distribution wherein the gaussian is in a scaled space.

    In the scaled space, the mean is multiplied by the inverse scale factor and then put
    into log space. This also applies to the bounds on the mean below.

    Keyword Arguments:
        scalefactor (np.ndarray): The scale factor for each gaussian distribution.

    Attributes:
        scalefactor (torch.Tensor): The scale factor for each gaussian distribution.
    """

    def __init__(
        self,
        *,
        initial_mean: ArrayLike,
        minmean: ArrayLike,
        maxmean: ArrayLike,
        initial_stddev: ArrayLike,
        minstddev: ArrayLike,
        maxstddev: ArrayLike,
        scalefactor: np.ndarray,
    ) -> None:
        self.scalefactor = torch.as_tensor(scalefactor)

        super().__init__(
            initial_mean=self.inversescale(torch.as_tensor(initial_mean)),
            minmean=self.inversescale(torch.as_tensor(minmean)),
            maxmean=self.inversescale(torch.as_tensor(maxmean)),
            initial_stddev=initial_stddev,
            maxstddev=maxstddev,
            minstddev=minstddev,
        )

    @property
    def invscalefactor(self) -> torch.Tensor:
        """torch.Tensor: The inverse scale factor for each gaussian distribution."""
        return 1 / self.scalefactor

    def inversescale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse scaling operation to x."""
        return torch.log(torch.multiply(self.invscalefactor, x))

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the scaling operation to x."""
        return torch.multiply(self.scalefactor, torch.exp(x))

    def sample(self) -> torch.Tensor:
        """Sample and return values in the scaled space."""
        return self.scale(self.unscaledsample())

    def unscaledsample(self) -> torch.Tensor:
        """Sample and return values in the unscaled space."""
        return self.distribution.rsample().detach().item()

    def logprob(self, x: torch.Tensor) -> torch.Tensor:
        """The log probability of the PDF at x.

        Arguments:
            x (torch.Tensor): A sample in the scaled space.

        Returns:
            torch.Tensor: The log probability.
        """
        y = self.inversescale(x)
        return self.distribution.log_prob(y)


# We don't make this a subclass of GaussianDistribution with stddev 0
# because torch.distributions.Normal doesn't allow stddev = 0
class DegenerateDistribution(Distribution):
    """A degenerate (deterministic) distribution.

    Keyword Arguments:
        initial_value (np.ndarray): The initial value of the distribution.
        minvalue (np.ndarray): The minimum value of the distribution.
        maxvalue (np.ndarray): The maximum value of the distribution.

    Attributes:
        initial_value (torch.Tensor): The initial value of the distribution.
        minvalue (torch.Tensor): The minimum value of the distribution.
        maxvalue (torch.Tensor): The maximum value of the distribution.
        value (torch.Tensor): The clamped value of the distribution.
    """

    def __init__(
        self,
        *,
        initial_value: np.ndarray,
        minvalue: np.ndarray,
        maxvalue: np.ndarray,
    ) -> None:
        super().__init__()
        self.initial_value = torch.as_tensor(initial_value)
        self.minvalue = torch.as_tensor(minvalue)
        self.maxvalue = torch.as_tensor(maxvalue)
        self._value = nn.parameter.Parameter(self.initial_value)

    @property
    def value(self) -> torch.Tensor:
        return torch.clip(self._value, min=self.minvalue, max=self.maxvalue)

    @property
    def mean(self) -> torch.Tensor:
        return self.value

    @property
    def stddev(self) -> torch.Tensor:
        return torch.zeros_like(self.value)

    @property
    def logstddev(self) -> torch.Tensor:
        return torch.log(self.stddev)

    @property
    def unclampedmean(self) -> torch.Tensor:  # type: ignore
        """torch.Tensor: Unclamped mean of the distribution."""
        return self._value

    @property
    def initial_mean(self) -> torch.Tensor:
        return self.initial_value

    @property
    def initial_stddev(self) -> torch.Tensor:
        return self.stddev

    def reset(self) -> None:
        self._value = nn.parameter.Parameter(self.initial_value)

    def sample(self) -> torch.Tensor:
        return self.value

    def logprob(self, _: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self._value)

    def entropy(self) -> torch.Tensor:
        return torch.zeros_like(self._value)

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Normal(
            loc=self.value, scale=torch.zeros_like(self.value)
        )


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
