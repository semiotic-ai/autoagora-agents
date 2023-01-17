# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import experiment

from .helper import add, sub


@pytest.fixture
def d():
    return {"add": add, "sub": sub}


def test_factory_executes_correct_function(d):
    n = "add"
    assert experiment.factory(n, d, 2, b=1, c=2) == 6


def test_factory_raises_notimplementederror(d):
    n = "mul"
    with pytest.raises(NotImplementedError):
        _ = experiment.factory(n, d, 2, b=1, c=2)
