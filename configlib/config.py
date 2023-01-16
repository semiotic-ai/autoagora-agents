# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import sacred


def experiment(*, name: str, spath: str, apath: str):
    """
    Create an experiment.

    Keyword Arguments:
        name (str): The name of the experiment
        spath (str): The path to the python file containing the simulation config
        apath (str): The path the python file containing the algorithm config

    Returns:
        sacred.Experiment: The constructed sacred experiment object.
    """
    ex = sacred.Experiment(name)
    names = ("simulation_ingredient", "algorithm_ingredient")
    paths = (spath, apath)
    ii = tuple(map(lambda n, p: importn(n, p), names, paths))
    return ex


def importn(n: str, p: str):
    """
    Import a given item from the python file at the specified path.

    Arguments:
        n (str): The name of the item to import
        p (str): The path to the python file containing the item to import.

    Returns:
        sacred.Ingredient: The imported ingredient from the specified path
    """
    # https://www.geeksforgeeks.org/how-to-import-a-python-module-given-the-full-path/
    spec = importlib.util.spec_from_file_location(n, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, n)
