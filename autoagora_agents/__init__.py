# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import sacred

algorithm_ingredient = sacred.Ingredient("algorithm")


@algorithm_ingredient.config
def config():
    foo = 100
    bar = 200
