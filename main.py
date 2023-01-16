# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

import configlib

parser = argparse.ArgumentParser(description="Run experiments for autoagora")
parser.add_argument("-n", "--name")
parser.add_argument("-s", "--simulation_path", default="simulation/__init__.py")
parser.add_argument("-a", "--algorithm_path", default="autoagora_agents/__init__.py")
args = parser.parse_args()

ex = configlib.experiment(
    name=args.name, spath=args.simulation_path, apath=args.algorithm_path
)


def main():
    print(args.name)
    print(ex.path)
    return None


if __name__ == "__main__":
    main()
