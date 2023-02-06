# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

import experiment
from simulation import environment, simulation_ingredient
from sacred import Experiment
from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # type: ignore

parser = argparse.ArgumentParser(description="Run experiments for autoagora")
parser.add_argument("-n", "--name")
parser.add_argument("-s", "--simulation_path", default="simulationconfig.py")
parser.add_argument("-a", "--algorithm_path", default="autoagora_agents/__init__.py")
args = parser.parse_args()

ex = experiment.experiment(
    name=args.name, spath=args.simulation_path, apath=args.algorithm_path
)


@ex.automain
def main():
    env = environment()  # type: ignore
