# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Any


def factory(n: str, d: dict[str, Callable], *args, **kwargs) -> Any:
    """Construct an object from the factory.

    Arguments:
        n (str): The name of associated with the function to call.
        d (dict[str, Callable]): A mapping between names and callables.

    Returns:
        Any: The value returned by the callable.

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


def decoratorfactoryhelper(*, kind: str, d: dict[str, Callable], **kwargs) -> Any:
    """Extract "kind" from the config.

    Keyword Arguments:
        kind (str): The kind of reward.
        d (dict[str, Reward]): A mapping between names and callables

    Returns:
        Any: The value returned by the callable.
    """
    return factory(kind, d, **kwargs)
