# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import experiment


def test_inbounds_float():
    a = np.array([1, 2, 3])
    l = 0
    h = 3
    assert experiment.inbounds(a, l, h)


def test_inbounds_array():
    a = np.array([1, 2, 3])
    l = np.array([0, 1, 2])
    h = np.array([2, 3, 4])
    assert experiment.inbounds(a, l, h)


def test_not_inbounds_float():
    a = np.array([1, 2, 3])
    l = 0
    h = 2
    assert not experiment.inbounds(a, l, h)


def test_not_inbounds_array():
    a = np.array([1, 2, 3])
    l = np.array([2, 3, 4])
    h = np.array([2, 3, 4])
    assert not experiment.inbounds(a, l, h)


def test_inbounds_empty():
    a = np.array([])
    l = 0
    h = 2
    assert experiment.inbounds(a, l, h)


def test_inbounds_raises_valueerror():
    a = np.array([])
    l = np.array([1, 2])
    h = 2
    with pytest.raises(ValueError):
        _ = experiment.inbounds(a, l, h)


def test_applybounds_float():
    a = np.array([0, 2, 4])
    l = 1
    h = 3
    assert experiment.inbounds(experiment.applybounds(a, l, h), l, h)


def test_applybounds_array():
    a = np.array([1, 2, 3])
    l = np.array([2, 3, 4])
    h = np.array([2, 3, 4])
    assert experiment.inbounds(experiment.applybounds(a, l, h), l, h)


def test_applybounds_empty():
    a = np.array([])
    l = 1
    h = 3
    assert experiment.inbounds(experiment.applybounds(a, l, h), l, h)


def test_applybounds_raises_valueerror():
    a = np.array([])
    l = np.array([1, 2])
    h = 3
    with pytest.raises(ValueError):
        _ = experiment.applybounds(a, l, h)
