Scripts
=======

multi_agent_simulation.py
--------------------------
Trains multiple agents based on the provided configuration file and plots agents and environment metrics. See simulation/configs folder for available configurations.

.. code:: bash

   poetry run python simulation/multi_agent_simulation.py -c simulation/configs/3different_agents_noisy_cyclic.json


show_simulated_subgraph.py
--------------------------

Plots the environment (queries/s as a function of price multiplier) and exports it into a mp4 movie.

.. code:: bash

   poetry run python simulation/show_simulated_subgraph.py 


show_bandit.py
---------------

Trains one of bandits on a selected simulated environment. Plots the agent policy (gaussian over a price multiplier) and environment (queries/s as a function of price multiplier)  and exports it into a mp4 movie.

.. code:: bash

   poetry run python simulation/show_bandit.py


train_bandit.py
---------------

Trains one of bandits on a selected simulated environment. Monitors various variables and logs them to Tensorboard.

.. code:: bash

   poetry run python simulation/train_bandit.py 
