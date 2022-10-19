[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Semantic Versioning](https://img.shields.io/badge/semver-2.0.0-green)](https://semver.org/spec/v2.0.0.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# AutoAgora Agents

## Developer's guide

### Installation directly from the source code

To install AutoAgora directly from the source code please clone the repository and install package in the virtual environment using `poetry`:
```console
git clone https://github.com/semiotic-ai/autoagora-agents.git
cd autoagora
poetry install
```

### Running the AutoAgora code

All scripts should be executed in the virtual environment managed by `poetry`.

### Running the test suite

```console
poetry run python -m pytest
```

### Running the bandit-related scripts

There are three scripts in the bandit_scripts folder:

- `show_simulated_subgraph.py` - runs and (optionally) visualizes the simulation (only environment)
- `show_bandit.py` - runs and (optionally) visualizes the simulation (both agent and environment)
- `train_bandit.py` - runs the simulation and trains an agent in a given environment (no visualization, logging to TensorBoard)

Each script can be parametrized by arguments passed from the command line. For example, to run the simulation with `ppo` agent in the `noisy_static` environment please run:

```console
poetry run python bandit_scripts/show_bandit.py -a ppo -e noisy_static --show
```

More details on arguments can be accessed in help (`--h`).
