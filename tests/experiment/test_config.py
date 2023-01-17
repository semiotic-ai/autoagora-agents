# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import experiment


def test_importn():
    assert experiment.config.importn("foo", "tests/experiment/helper.py") == "foo"


# TODO: Need to specify what default configs look like first
@pytest.mark.skip
def test_experiment():
    assert True
