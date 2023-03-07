# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import experiment


def inbounds(a: np.ndarray, l: float | np.ndarray, h: float | np.ndarray) -> np.bool_:
    """Check if array is in between lower and upper bounds.

    Bounds are inclusive.

    Arguments:
        a (np.ndarray): The array to check
        l (float | np.ndarray): The lower bound
        h (float | np.ndarray): The upper bound

    Returns:
        bool: True if array is in bounds, else False.

    Raises:
        ValueError: If length of the bounds don't equal the length of the array, if the
            bounds are given by arrays.
    """
    return ((a >= l) & (a <= h)).all()


def test_applybounds_float():
    a = np.array([0, 2, 4])
    l = 1
    h = 3
    assert inbounds(experiment.applybounds(a, l, h), l, h)


def test_applybounds_array():
    a = np.array([1, 2, 3])
    l = np.array([2, 3, 4])
    h = np.array([2, 3, 4])
    assert inbounds(experiment.applybounds(a, l, h), l, h)


def test_applybounds_empty():
    a = np.array([])
    l = 1
    h = 3
    assert inbounds(experiment.applybounds(a, l, h), l, h)


def test_applybounds_raises_valueerror():
    a = np.array([])
    l = np.array([1, 2])
    h = 3
    with pytest.raises(ValueError):
        _ = experiment.applybounds(a, l, h)
