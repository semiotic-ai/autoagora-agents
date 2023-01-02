from asyncio import run
from typing import List

import pytest

from environments.simulated_subgraph import NoisyCyclicQueriesSubgraph
from simulation.controller import set_random_seed


def test_random_seed():
    """Test consistent random number generation based on seed"""
    set_random_seed(42)

    # Noisy enviroments uses numpy random for noise generation
    env = NoisyCyclicQueriesSubgraph()

    # Set cost multiplier.
    run(env.set_cost_multiplier(1e-6))

    qps1 = run(env.queries_per_second())
    assert qps1 == 0.5124178538252808

    qps2 = run(env.queries_per_second())
    assert qps2 == 0.4965433924707204
