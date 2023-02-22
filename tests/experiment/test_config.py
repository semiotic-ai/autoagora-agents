# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import experiment


def test_importn():
    assert experiment.config.importn("foo", "tests/experiment/helper.py") == "foo"
