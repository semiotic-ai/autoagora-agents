# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.typing import NDArray


def inbounds(a: NDArray, l: float | NDArray, h: float | NDArray) -> np.bool_:
    """Check if array is in between lower and upper bounds.

    Bounds are inclusive.

    Arguments:
        a (NDArray): The array to check
        l (float | NDArray): The lower bound
        h (float | NDArray): The upper bound

    Returns:
        bool: True if array is in bounds, else False.

    Raises:
        ValueError: If length of the bounds don't equal the length of the array, if the
            bounds are given by arrays.
    """
    return ((a >= l) & (a <= h)).all()


def applybounds(a: NDArray, l: float | NDArray, h: float | NDArray) -> NDArray:
    """Set out of bounds values to be between bounds.

    Bounds are inclusive.

    Arguments:
        a (NDArray): The array to which to apply bounds.
        l (float | NDArray): The lower bound
        h (float | NDArray): The upper bound

    Returns:
        NDArray: The input array with the out of bounds values set to be in bounds.

    Raises:
        ValueError: If length of the bounds don't equal the length of the array, if the
            bounds are given by arrays.
    """
    return np.minimum(np.maximum(a, l), h)
