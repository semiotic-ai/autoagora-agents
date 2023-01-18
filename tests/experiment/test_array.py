# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import experiment


def test_inbounds():
    a = np.array([1, 2, 3])
    l = 0
    h = 3
    assert experiment.inbounds(a, l, h)


def test_inbounds_failure():
    a = np.array([1, 2, 3])
    l = 0
    h = 2
    assert not experiment.inbounds(a, l, h)
