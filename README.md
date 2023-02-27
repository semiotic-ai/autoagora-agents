[![Coveralls](https://img.shields.io/coveralls/github/semiotic-ai/autoagora-agents)](https://coveralls.io/github/semiotic-ai/autoagora-agents)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Semantic Versioning](https://img.shields.io/badge/semver-2.0.0-green)](https://semver.org/spec/v2.0.0.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://img.shields.io/github/actions/workflow/status/semiotic-ai/autoagora-agents/gh-pages.yml?label=docs)](https://semoitic-ai.github.io/autoagora-agents/)

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

### Running Experiments

Currently, to run an experiment, you should specify three components: `--name` (`-n`), `--simulation_path` (`-s`), and `--algorithm_path` (`-a`).
The `--name` field is just the name to give to the experiment. 
Once we set up a Mongo Observer to track experiments with a MongoDB, this will help you track different experiments more efficiently.
The `--simulation_path` and `--algorithm_path` fields should point to a simulation configuration file and an algorithm configuration file, respectively.

**Note:** We do not use default values _anywhere_ in our code.
Every value must come from the config file.

#### Simulation Config

The simulation config must be a python file.
Let's break down the various components of the simulation config.

The first thing you need to do is define a function that is captured by the simulation ingredient.
The name of the function itself doesn't matter, but we use `config` here for clarity.

``` python
from simulation import simulation_ingredient  # Import the simulation ingredient

@simulation_ingredient.config  # Tag as the simulation config
def config():
    ...
```

We use the simulation config to construct the simulation `Environment`.
As such, the config should specify the inputs to the `Environment` class' initialiser.

``` python
"""
    isa (dict[str, Any]): The config for the ISA.
    entities (list[dict[str, Any]]): The configs for each group of entities.
    nepisodes (int): How many episodes to run.
    ntimesteps (int): How many timesteps to run each episode for.
"""
```

`nepisodes` and `ntimesteps` are self-explanatory.

`isa` is a dictionary that specifies the configuration of which ISA to use.
See the `isa` documentation for more details.

`entities` is a list of configs for each entity type.
Each entry in the `entities` list is a dictionary.
The dictionary can be of kind `entity` in which case the entity has only a state, but no action, or `agent` in which case the entity has a state and an action.
Check out the `entity` documentation for more details.

#### Algorithm Config

Similarly to the simulation config, the first thing you need to do is define a function that is captured by the algorithm ingredient.

``` python
from autoagora_agents import algorithm_ingredient  # Import the algorithm ingredient

@algorithm_ingredient.config  # Tag as the algorithm config
def config():
    ...
```

The experiment uses the algorithm config to construct the `Controller` object, which maps the simulation to the algorithms.
The `Controller` also constructs the agents.
It takes two inputs: `seed` and `agents`.

The `seed` is just the random seed set for reproducibility.

The `agents` entry is a list of dictionaries similar to `entities` from the simulation config.
In fact, `agents` does much the same for the algorithm side of the code as `entities` does for the simulation side of the code.
In it, each entry is a dictionary specifying the configuration of an algorithm for a particular group.
Note here that the `"group"` field must match the `"group"` field of an agent type in the `entities` list.
This is how the code knows how to map between the simulation and the algorithm.
Other than that, the rest of the dictionary should map to the configuration of a particular algorithm.
See the `algorithm` documentation for more details.
