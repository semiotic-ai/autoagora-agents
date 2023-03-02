# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

from sacred import SETTINGS

import experiment
from autoagora_agents import controller
from simulation import environment

# For good reason, sacred disallows modifying your config file in the code.
# However, our code does some clever stuff to make configs less verbose than they'd
# otherwise need to be, so we disable this check
SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # type: ignore

parser = argparse.ArgumentParser(description="Run experiments for autoagora")
parser.add_argument("-n", "--name")
parser.add_argument("-s", "--simulation_path", default="simulationconfig.py")
parser.add_argument("-a", "--algorithm_path", default="algorithmconfig.py")
args = parser.parse_args()

ex = experiment.experiment(
    name=args.name, spath=args.simulation_path, apath=args.algorithm_path
)


@ex.automain
def main():
    # NOTE: The structure of this loop is very bandit-specific.
    # This would not work for a more complex RL algorithm without
    # modifications
    algs = controller()  # type: ignore
    env = environment()  # type: ignore
    for _ in range(env.nepisodes):
        obs, act, rew, done = env.reset()
        while not env.isfinished():
            act = algs(observations=obs, actions=act, rewards=rew, dones=done)
            algs.update()
            obs, act, rew, done = env.step(actions=act)
