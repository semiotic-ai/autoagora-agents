# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import sacred


def experiment(*, name: str, spath: str, apath: str, epath: str):
    """Create an experiment.

    Keyword Arguments:
        name (str): The name of the experiment
        spath (str): The path to the python file containing the simulation config
        apath (str): The path the python file containing the algorithm config
        epath (str): The path the python file containing the experiment config

    Returns:
        sacred.Experiment: The constructed sacred experiment object.
    """
    names = ("simulation_ingredient", "algorithm_ingredient", "experiment_ingredient")
    paths = (spath, apath, epath)
    ii = tuple(map(lambda n, p: importn(n, p), names, paths))
    ex = sacred.Experiment(name, ingredients=ii)
    return ex


def importn(n: str, p: str):
    """Import a given item from the python file at the specified path.

    Arguments:
        n (str): The name of the item to import
        p (str): The path to the python file containing the item to import.

    Returns:
        sacred.Ingredient: The imported ingredient from the specified path
    """
    # https://www.geeksforgeeks.org/how-to-import-a-python-module-given-the-full-path/
    spec = importlib.util.spec_from_file_location(n, p)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, n)
