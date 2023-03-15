# Running Experiments

We are often interested in running experiments to understand the performance of various agents.
We can set up experiments by specifying three configuration files:

* _experimentconfig.py_
* _simulationconfig.py_
* _algorithmconfig.py_

Let's take each of these in turn.

## Experiment Configuration

`autoagora_agents` uses [sacred](https://github.com/IDSIA/sacred) to run and track experiments.
This has implications for how you should construct configuration files.
Take a look at the following example configuration.

```python 
from experiment import experiment_ingredient

@experiment_ingredient.config
def config():
    seed = 0
```

Let's break this down.
To begin with, you must associate each configuration with an `ingredient`.
In this case, we use the `experiment_ingredient` from `experiment`.
We decorate a function, here named `config()` although the name itself doesn't matter, with `@experiment_ingredient.config`.
This informs `sacred` that this function defines the configuration for the `experiment_ingredient`.
In general, the best way to remember this is to simply copy whatever `.py` file you wish to modify and then modify the values you wish to change.
This way, you shouldn't need to worry about the boilerplate we just discussed.

Moving on, let's now discussion the various configuration parameters for the `experiment_ingredient`.

* `seed (int)`: The random seed of the experiment ensures that the experiment is reproducible.

## Simulation Configuration

For our experiments, we also need to set up simulations that could result in interesting behaviour.
Here, we have the `simulation_ingredient` with its corresponding configuration from *simulationconfig.py*.
Notice that this file has the same form as the *experimentconfig.py* file had.
We import the `simulation_ingredient` and then mark a function with the `@simulation_ingredient.config` decorator.

```python
import numpy as np
from simulation import simulation_ingredient

@simulation_ingredient.config
def config():
    nproducts = 1  # Not actually part of the configuration, but we specify this as a helper
    ntimesteps = 10000
    nepisodes = 1
    distributor = {"kind": "softmax", "source": "consumer", "to": "indexer"}
    entities = [
        {
            "kind": "entity",
            "count": 1,
            "group": "consumer",
            "state": {
                "kind": "budget",
                "low": 0,
                "high": 1,
                "initial": 0.5 * np.ones(nproducts),
                "traffic": np.ones(nproducts),
            },
        },
        {
            "kind": "agent",
            "count": 1,
            "group": "indexer",
            "state": {
                "kind": "price",
                "low": np.zeros(nproducts),
                "high": 3 * np.ones(nproducts),
                "initial": np.ones(nproducts),
            },
            "action": {
                "kind": "price",
                "low": np.zeros(nproducts),
                "high": 3 * np.ones(nproducts),
                "shape": (nproducts,),
            },
            "reward": [
                {
                    "kind": "traffic",
                    "multiplier": 1,
                }
            ],
            "observation": [
                {
                    "kind": "bandit",
                }
            ],
        },
    ]
```

Let's break down this file's parts in greater detail.
To begin with, you should note that the config here is more-or-less directly passed into the `Environment`, so you can always reference that section of the reference documentation for greater detail.

* `ntimesteps (int)`: The number of timesteps for which each episode runs.
* `nepisodes (int)`: The number of episodes for which the simulation runs.
* `distributor (dict)`: The configuration for the distributor.
The specific details of this dictionary will depend on which distributor you wish to use in your simulation, so refer to that part of the `Reference` if you're interested in modifying that.
* `entities (list[dict])`: A list of the configuration for each group of entities in the simulation. Let's break this one down in a little more detail.

### Entities

Entities are *things within the environment*.
They have a state, but that's about it.
An example of a case in which you might create an entity group would be for some sort of static consumer, which always returns the same budget.

* `kind (str)`: `"entity"`
* `count (int)`: How many of these entities there are in the environment.
* `group (str)`: This labels all the entities in this group with a common name, both for experiment tracking and for simulation internals. Make sure each group of entities has a unique name.
* `state (dict)`: A configuration dictionary for the state of the entity.
See the reference documentation for details

Agents, on the other hand, are a subset of entities.
Think of it as analogous to how all squares are rectangles, but not all rectangles are squares.
All agents are entities, but not all entities are agents.
Agents are entities that have an associated algorithmic policy to guide their actions.
As a result, since agents are more typical RL structures, agents need to represent an MDP.
Thus, agents have not only a state, but also an action, a reward, and an observation.

* `kind (str)`: `"agent"`
* `count (int)`: How many of these agents there are in the environment.
* `group (str)`: This labels all the agents in this group with a common name, both for experiment tracking and for simulation internals. Make sure each group of agents has a unique name.
* `state (dict)`: A configuration dictionary for the state of the agents.
See the reference documentation for details
* `action (dict)`: A configuration dictionary for the action of the agents.
See the reference documentation for details.
* `reward (list[dict])`: Each element in this list is the configuration of a particular part of the reward function.
When computing the reward, the code will sum the different components, so if you want to use a cost term, make sure to set the `multiplier` parameter to be negative for the relevant cost terms.
See the reference documentation for details on different reward concretions.
* `observation (list[dict])`: This is similar to the rewards in that you specify a list of configurations.
Rather than summing, as is the case for the reward, the code will concatenate each part of your observation into a single vector.
See the reference documentation for details on different observation concretions.

## Algorithm

For historical reasons, the `algorithm_ingredient` comes from `autoagora_agents` (rather than `algorithm` as would be keeping with the pattern).
Let's take a look at the following from *algorithmconfig.py*.

```python
from autoagora_agents import algorithm_ingredient

@algorithm_ingredient.config
def config():
    agents = [
        {
            "kind": "ppobandit",
            "group": "indexer",
            "count": 1,
            "bufferlength": 10,
            "actiondistribution": {
                "kind": "gaussian",
                "initial_mean": [0.1],
                "initial_stddev": [0.1],
                "minmean": [0.0],
                "maxmean": [2.0],
                "minstddev": [1e-10],
                "maxstddev": [1.0],
            },
            "optimizer": {"kind": "sgd", "lr": 0.01},
            "ppoiterations": 2,
            "epsclip": 0.01,
            "entropycoeff": 1.0,
            "pullbackstrength": 0.0,
            "stddevfallback": True,
        }
    ]
```

The algorithm configuration contains the list `agents`.
Each entry in the `agents` list in the algorithm config must correspond with an entity of kind `agent` in the simulation config.
In fact, they must even share the same `"group"` so that the code knows how to route information between the simulation and algorithm.
In this case, we can see that the one entry in the list has group `"indexer"`, which corresponds to the one agent group in the simulation config above!
The counts must also match.
Other than that, we refer you to the algorithm refrence documentation for the specific details of what is in each agent's dictionary as this will vary from algorithm to algorithm.
The common parts are:

* `kind (str)`: The type of algorithm this agent runs.
* `group (str)`: Must correspond to an agent group in the simulation.
* `count (int)`: Must correspond to the count of the corresponding agent group in the simulation.


## Running Your Experiment

You've now configured your experiment.
To run it, just run: `poetry run python main.py`
