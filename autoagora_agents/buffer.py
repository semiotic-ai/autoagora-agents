# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

from jax import numpy as jnp


def buffer(*, maxlength: int) -> deque[dict[str, jnp.ndarray]]:
    """Create a buffer.

    Keyword Arguments:
        maxlength (int): The maximum length of the buffer.

    Returns:
        deque[dict[str, jnp.ndarray]]: The empty buffer.
    """
    b: deque[dict[str, jnp.ndarray]] = deque(maxlen=maxlength)
    return b


def isfull(b: deque[dict[str, jnp.ndarray]]) -> bool:
    """Return true if the buffer is full. Else false."""
    return len(b) == b.maxlen


def get(k: str, b: deque[dict[str, jnp.ndarray]]) -> jnp.ndarray:
    """Get key from elements of the buffer.

    Arguments:
        k (str): The key.
        b (deque[dict[str, jnp.ndarray]]): The empty buffer.

    Returns:
        ArrayList: The matching elements
    """
    return jnp.array([_b[k] for _b in b])
