# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp

from autoagora_agents import buffer


def test_buffer():
    maxlen = 10

    b = buffer.buffer(maxlength=maxlen)
    sample = {"reward": jnp.array([1, 2, 3]), "action": jnp.array([3, 2, 1])}
    assert len(b) == 0
    b.append(sample)  # type: ignore
    assert len(b) == 1

    for _ in range(maxlen + 1):
        b.append(sample)  # type: ignore
    assert buffer.isfull(b)

    b.clear()
    assert not buffer.isfull(b)
