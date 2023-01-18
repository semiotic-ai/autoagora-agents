# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import experiment


def test_inbounds():
    a = np.array([1, 2, 3])
    l = 0
    h = 3
    assert experiment.inbounds(a, l, h)


def test_not_inbounds():
    a = np.array([1, 2, 3])
    l = 0
    h = 2
    assert not experiment.inbounds(a, l, h)


def test_inbounds_empty():
    a = np.array([])
    l = 0
    h = 2
    assert experiment.inbounds(a, l, h)


def test_applybounds():
    a = np.array([0, 2, 4])
    l = 1
    h = 3
    assert experiment.inbounds(experiment.applybounds(a, l, h), l, h)


def test_applybounds_empty():
    a = np.array([])
    l = 1
    h = 3
    assert experiment.inbounds(experiment.applybounds(a, l, h), l, h)
