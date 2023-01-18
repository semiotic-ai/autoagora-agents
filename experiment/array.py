# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def inbounds(a: np.ndarray, l: float, h: float) -> np.bool_:
    """Check if array is in between lower and upper bounds.

    Bounds are inclusive.

    Arguments:
        a (np.ndarray): The array to check
        l (float): The lower bound
        h (float): The upper bound
    """
    return ((a >= l) & (a <= h)).all()
