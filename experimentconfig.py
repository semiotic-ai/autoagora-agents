# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from experiment import experiment_ingredient


@experiment_ingredient.config
def config():
    seed = 0
