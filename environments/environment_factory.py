# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Union

from environments.shared_subgraph import NoisySharedSubgraph
from environments.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyCyclicZeroQueriesSubgraph,
    NoisyDynamicQueriesSubgraph,
    NoisyQueriesSubgraph,
)

_ENVIRONMENT_TYPES = {
    "NoisyQueriesSubgraph": NoisyQueriesSubgraph,
    "static": NoisyQueriesSubgraph,
    "noisy_static": NoisyQueriesSubgraph,
    "NoisyCyclicQueriesSubgraph": NoisyCyclicQueriesSubgraph,
    "cyclic": NoisyCyclicQueriesSubgraph,
    "noisy_cyclic": NoisyCyclicQueriesSubgraph,
    "NoisyCyclicZeroQueriesSubgraph": NoisyCyclicZeroQueriesSubgraph,
    "cyclic_zero": NoisyCyclicZeroQueriesSubgraph,
    "noisy_cyclic_zero": NoisyCyclicZeroQueriesSubgraph,
    "NoisyDynamicQueriesSubgraph": NoisyDynamicQueriesSubgraph,
    "dynamic": NoisyDynamicQueriesSubgraph,
    "noisy_dynamic": NoisyDynamicQueriesSubgraph,
    "NoisySharedSubgraph": NoisySharedSubgraph,
    "noisy_shared": NoisySharedSubgraph,
    "shared": NoisySharedSubgraph,
}


class EnvironmentFactory(object):
    """Factory creating environments.

    Args:
        environment_type: Type of the environment (Options: "static", "noisy_static", "cyclic", "noisy_cyclic")
        args: List of arguments passed to agent constructor.
        kwargs: Dict of keyword arguments passed to agent constructor.
    """

    def __new__(
        cls, environment_type: str, *args, **kwargs
    ) -> Union[NoisyQueriesSubgraph, NoisyQueriesSubgraph]:
        # If argument is set - do nothing.
        if "noise" not in kwargs.keys():
            # If not, try to extract "noise" value from the name.
            if "noisy" in environment_type:
                kwargs["noise"] = True
            else:
                kwargs["noise"] = False
        # Create the environment object.
        return _ENVIRONMENT_TYPES[environment_type](*args, **kwargs)


def add_environment_argparse(parser: argparse.ArgumentParser):
    """Adds argparse arguments related to environment to parser."""
    parser.add_argument(
        "-e",
        "--environment",
        default="noisy_cyclic",
        help="Sets the environment type (DEFAULT: noisy_cyclic)",
    )