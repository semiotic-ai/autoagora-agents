# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def applybounds(
    a: np.ndarray, l: float | np.ndarray, h: float | np.ndarray
) -> np.ndarray:
    """Set out of bounds values to be between bounds.

    Bounds are inclusive.

    Arguments:
        a (np.ndarray): The array to which to apply bounds.
        l (float | np.ndarray): The lower bound
        h (float | np.ndarray): The upper bound

    Returns:
        np.ndarray: The input array with the out of bounds values set to be in bounds.

    Raises:
        ValueError: If length of the bounds don't equal the length of the array, if the
            bounds are given by arrays.
    """
    return np.minimum(np.maximum(a, l), h)
