# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable


def factory(n: str, d: dict[str, Callable], *args, **kwargs) -> object:
    """
    Construct an object from the factory.

    Arguments:
        n (str): The name of object to create.
        d (dict[str, Callable]): A mapping between names and object constructors.

    Returns:
        object: The constructed object.

    Raises:
        NotImplementedError: If the dictionary does not contain the requested object constructor.
    """
    try:
        o = d[n](*args, **kwargs)
    except KeyError:
        raise NotImplementedError(
            f"The requested type {n} has not yet been added to the factory."
        )

    return o
