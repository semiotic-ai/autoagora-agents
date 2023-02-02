# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

from jax._src.typing import ArrayLike


def buffer(*, maxlength: int) -> deque[dict[str, ArrayLike]]:
    """Create a buffer.

    Keyword Arguments:
        maxlength (int): The maximum length of the buffer.

    Returns:
        deque[dict[str, ArrayLike]]: The empty buffer.
    """
    b: deque[dict[str, ArrayLike]] = deque(maxlen=maxlength)
    return b


def isfull(b: deque[dict[str, ArrayLike]]) -> bool:
    return len(b) == b.maxlen
