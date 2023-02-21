# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from collections import deque

import torch


def buffer(*, maxlength: int) -> deque[dict[str, Any]]:
    """Create a buffer.

    Keyword Arguments:
        maxlength (int): The maximum length of the buffer.

    Returns:
        deque[dict[str, Any]]: The empty buffer.
    """
    b: deque[dict[str, Any]] = deque(maxlen=maxlength)
    return b


def isfull(b: deque[dict[str, torch.Tensor]]) -> bool:
    """Return true if the buffer is full. Else false."""
    return len(b) == b.maxlen


def get(k: str, b: deque[dict[str, Any]]) -> torch.Tensor:
    """Get key from elements of the buffer.

    Arguments:
        k (str): The key.
        b (deque[dict[str, Any]]): The empty buffer.

    Returns:
        torch.TensorList: The matching elements
    """
    return torch.as_tensor([_b[k] for _b in b])
