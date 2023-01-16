# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import sacred

simulation_ingredient = sacred.Ingredient("simulation")


@simulation_ingredient.config
def config():
    a = 2
    b = "c"
