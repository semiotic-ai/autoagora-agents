# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from asyncio import run
import time
from typing import Mapping


import ffmpeg
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from environments.simulated_subgraph import SimulatedSubgraph
from simulation.chart import ChartsWidget,ChartWidget
from simulation.controller import init_simulation
from simulation.show_bandit import add_experiment_argparse

logging.basicConfig(level="WARN", format="%(message)s")

LOG_PLOT = True
WINDOW_SIZE = (1000, 1000)

async def main():
    # Init argparse.
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-c ...] [-i ...] [--show] [--save]",
        description="Runs multi-agent simulation and (optionally) shows it and/or saves it to a file.",
    )
    add_experiment_argparse(parser=parser)

    # Initialize the simulation.
    args, environment, agents = init_simulation(parser=parser)
    # We need the environment to be SimulatedSubgraph
    assert isinstance(environment, SimulatedSubgraph)

    # Environment x.
    min_x = 1e-8
    max_x = 8e-5
    
    save_file = f"{args.config}.mp4" if args.save else None
    
    charts = ChartsWidget(save_file=save_file,size=WINDOW_SIZE,title="Multi-agent training",default_chart_config={"chart_height":200,"pen":{"width":1.5},"autoDownsample":True})
    
    policy_chart = charts.addChart("policy",
        title="time 0",
        config={
            "chart_height":300,
            "x_axis":{
                "label":"Price multiplier",
                "log":True
            },
            "y_axis":{
                "label":"Query rate",
                "range":(0,1.3)
            }
        }
    )
    
    query_rate_chart = charts.addChart("query_rate",
        config={
            "x_axis":{
                "label":"Timestep",
            },
            "y_axis":{
                "label":"Query rate",
            }
        }
    )
    
    query_total_chart = charts.addChart("query_total",
        config={
            "x_axis":{
                "label":"Timestep",
            },
            "y_axis":{
                "label":"Total queries",
            }
        }
    )
    
    revenue_rate_chart = charts.addChart("revenue_rate",
        config={
            "x_axis":{
                "label":"Timestep",
            },
            "y_axis":{
                "label":"Revenue rate",
            }
        }
    )
    
    revenue_total_chart = charts.addChart("revenue_total",
        config={
            "x_axis":{
                "label":"Timestep",
            },
            "y_axis":{
                "label":"Total revenue",
            }
        }
    )        
    
    for i,agent_name in enumerate(agents.keys()):
        
        policy_chart.addPlot(f'a{i}',f"Agent {agent_name}: policy",config={"pen":{"color":(i, len(agents) + 1)}})
        policy_chart.addPlot(f'i{i}',f"Agent {agent_name}: init policy",config={"pen":{"color":(i, len(agents) + 1),"style":QtCore.Qt.DotLine}})
        policy_chart.addPlot(f'q{i}',f"Agent {agent_name}: query rate",config={"pen":None,"symbolBrush":(i, len(agents) + 1),"symbolPen":"w"})
        
        query_rate_chart.addPlot(f'a{i}',f"Agent {agent_name}",config={"pen":{"color":(i, len(agents) + 1)}})
        
        query_total_chart.addPlot(f'a{i}',f"Agent {agent_name}",config={"pen":{"color":(i, len(agents) + 1)}})
        
        revenue_rate_chart.addPlot(f'a{i}',f"Agent {agent_name}",config={"pen":{"color":(i, len(agents) + 1)}})
        
        revenue_total_chart.addPlot(f'a{i}',f"Agent {agent_name}",config={"pen":{"color":(i, len(agents) + 1)}})
        
    policy_chart.addPlot("e","Environment: total query rate",config={"pen":{"color":"grey"}})

    query_total_chart.addPlot("d","Dropped",config={"pen":{"color":(len(agents), len(agents) + 1)}})
    
    queries_per_second = [[] for _ in agents]
    revenue_rate_data = [[] for _ in agents]
    total_revenue_data = [[] for _ in agents]
    total_agent_queries_data = [[] for _ in agents]
    total_unserved_queries_data = []

    for i in range(args.iterations):
        logging.debug("=" * 20 + " step %s " + "=" * 20, i)

        # X. Visualize the environment.
        if i % args.fast_forward_factor == 0:
            # Plot environment.
            env_x, env_y = await environment.generate_plot_data(min_x, max_x)
            policy_chart.setPlotData("e",env_x, env_y)

        # Execute actions for all agents.
        scaled_bids = []
        for agent_id, (agent_name, agent) in enumerate(agents.items()):
            # 1. Get bid from the agent (action)
            scaled_bids.append(agent.get_action())
            if agent_id == 0:
                logging.debug("Agent %s action: %s", agent_id, scaled_bids[agent_id])

            # 2. Act: set multiplier in the environment.
            await environment.set_cost_multiplier(
                    scaled_bids[agent_id], agent_id=agent_id
                )
     

        # Make a step. (Executes a number of queries in the case of the ISA)
        environment.step()

        # Get observations for all agents.
        for agent_id, (agent_name, agent) in enumerate(agents.items()):
            # 3. Get the rewards.
            # Get queries per second for a given .
            queries_per_second[agent_id] += [
               await environment.queries_per_second(agent_id=agent_id)
            ]
            # Turn it into "monies".
            monies_per_second = queries_per_second[agent_id][-1] * scaled_bids[agent_id]
            # Add reward.
            agent.add_reward(monies_per_second)

            revenue_rate_data[agent_id] += [monies_per_second]
            if i > 0:
                total_revenue_data[agent_id] += [
                    total_revenue_data[agent_id][-1] + revenue_rate_data[agent_id][-1]
                ]
            else:
                total_revenue_data[agent_id] += [revenue_rate_data[agent_id][-1]]

            # 4. Update the policy.
            if True:  # agent_id == 0:
                if hasattr(agent, "reward_buffer"):
                    logging.debug(
                        "Agent %s reward_buffer = %s",
                        agent_id,
                        agent.reward_buffer,
                    )
                    logging.debug(
                        "Agent %s action_buffer = %s",
                        agent_id,
                        agent.action_buffer,
                    )
                if hasattr(agent, "mean"):
                    logging.debug(
                        "Agent %s mean = %s",
                        agent_id,
                        agent.mean(),
                    )
                    logging.debug(
                        f"Agent %s stddev = %s",
                        agent_id,
                        agent.stddev(),
                    )
                    logging.debug(
                        f"Agent %s initial_mean = %s",
                        agent_id,
                        agent.mean(initial=True),
                    )

                logging.debug(
                    "Agent %s observation: %s",
                    agent_id,
                    queries_per_second[agent_id][-1],
                )
            loss = agent.update_policy()
            logging.debug(f"Agent %s loss = %s", agent_id, loss)

            # Agents total queries served (for the plots)
            if i > 0:
                total_agent_queries_data[agent_id] += [
                    total_agent_queries_data[agent_id][-1]
                    + queries_per_second[agent_id][-1]
                ]
            else:
                total_agent_queries_data[agent_id] += [queries_per_second[agent_id][-1]]

        # Total unserved queries
        if i > 0:
            total_unserved_queries_data += [
                total_unserved_queries_data[-1]
                + 1
                - sum(e[-1] for e in queries_per_second)
            ]
        else:
            total_unserved_queries_data += [1 - sum(e[-1] for e in queries_per_second)]

        # X. Collect the values for visualization of agent's gaussian policy.
        if i % args.fast_forward_factor == 0:
            for agent_id, (agent_name, agent) in enumerate(agents.items()):
                
                agent_key = f'a{agent_id}'
                
                # Get data.
                data = await agent.generate_plot_data(min_x, max_x, logspace=LOG_PLOT)
                agent_x = data.pop("x")
                agent_y = data["policy"]
                policy_chart.setPlotData(agent_key,agent_x, agent_y)
                query_rate_chart.setPlotData(agent_key,queries_per_second[agent_id])
               
               # Plot init policy and add it to last list in container.
                if "init policy" in data.keys():
                    init_agent_y = data["init policy"]
                    policy_chart.setPlotData(f'i{agent_id}',agent_x, init_agent_y)

                # Agent q/s.
                agent_qps_x = min(max_x, max(min_x, scaled_bids[agent_id]))
                
                policy_chart.setPlotData(
                    f'q{agent_id}',
                    [agent_qps_x], 
                    [queries_per_second[agent_id][-1]]
                )
                
                # Total queries served by agent
                query_total_chart.setPlotData(agent_key,total_agent_queries_data[agent_id])
                
                # Revenue rate by agent
                revenue_rate_chart.setPlotData(agent_key,revenue_rate_data[agent_id])
                
                # Total revenue by agent
                revenue_total_chart.setPlotData(agent_key,total_revenue_data[agent_id])

            # Total queries unserved
            query_total_chart.setPlotData("d",total_unserved_queries_data)

            policy_chart.title = f"time {i}"

            charts.render()

if __name__ == "__main__":
    run(main())
