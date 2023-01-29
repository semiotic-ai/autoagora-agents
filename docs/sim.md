# Simulation

## Controller

::: simulation.controller
    handler: python
    options:
        show_root_heading: true
        heading_level: 3

## Chart

::: simulation.chart
    handler: python
    options:
        show_root_heading: true
        heading_level: 3

## Scripts

### show_simulated_subgraph.py

Plots the environment (queries/s as a function of price multiplier) and exports it into a mp4 movie.

```bash
   poetry run python bandit_scripts/show_simulated_subgraph.py 
```

### show_bandit.py

Trains one of bandits on a selected simulated environment. Plots the agent policy (gaussian over a price multiplier) and environment (queries/s as a function of price multiplier)  and exports it into a mp4 movie.

```bash
   poetry run python bandit_scripts/show_bandit.py
```

### train_bandit.py

Trains one of bandits on a selected simulated environment. Monitors various variables and logs them to Tensorboard.

```bash
   poetry run python bandit_scripts/train_bandit.py 
```