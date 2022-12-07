# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from asyncio import run

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from environments.simulated_subgraph import SimulatedSubgraph
from simulation.chart import (
    add_line_plot,
    add_scatter_plot,
    close_video_process,
    create_chart,
    create_layout,
    create_video_process,
    render_video_frame,
)
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

    ffmpeg_process = None

    # Environment x.
    min_x = 1e-8
    max_x = 8e-5

    layout = create_layout(
        WINDOW_SIZE,
        "Multi-agent training",
        not args.save,
        antialias=True,
        foreground="white",
    )

    policy_chart = create_chart(
        layout,
        title="time 0",
        height=300,
        legend_width=300,
        x_label="Price multiplier",
        y_label="Query rate",
        x_log=True,
        y_range=(0, 1.3),
    )

    # Policy PD
    agents_dist = [
        add_line_plot(
            policy_chart,
            f"Agent {agent_name}: policy",
            color=(i, len(agents) + 1),
            width=1.5,
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    # Initial policy PD
    agents_init_dist = [
        add_line_plot(
            policy_chart,
            f"Agent {agent_name}: init policy",
            color=(i, len(agents) + 1),
            width=1.5,
            style=QtCore.Qt.DotLine,  # type: ignore
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    # This is a line plot with invisible line and visible data points.
    # Easier to scale with the rest of the plot than with using a ScatterPlot.
    agents_scatter_qps = [
        add_scatter_plot(
            policy_chart,
            f"Agent {agent_name}: query rate",
            color=(i, len(agents) + 1),
            border="w",
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    # Environment QPS
    env_plot = add_line_plot(
        policy_chart, "Environment: total query rate", color="grey", width=1.5
    )

    query_rate_chart = create_chart(
        layout, legend_width=300, x_label="Timestep", y_label="Query rate"
    )

    agent_qps_plots = [
        add_line_plot(
            query_rate_chart,
            f"Agent {agent_name}",
            color=(i, len(agents) + 1),
            width=1.5,
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    queries_per_second = [[] for _ in agents]

    total_queries_chart = create_chart(
        layout, legend_width=300, x_label="Timestep", y_label="Total queries"
    )

    total_agent_queries_plots = [
        add_line_plot(
            total_queries_chart,
            f"Agent {agent_name}",
            color=(i, len(agents) + 1),
            width=1.5,
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    total_unserved_queries_plot = add_line_plot(
        total_queries_chart, "Dropped", color=(len(agents), len(agents) + 1), width=1.5
    )

    total_agent_queries_data = [[] for _ in agents]
    total_unserved_queries_data = []

    # Create revenue rate plot
    revenue_rate_chart = create_chart(
        layout, legend_width=300, x_label="Timestep", y_label="Revenue rate"
    )

    revenue_rate_plots = [
        add_line_plot(
            revenue_rate_chart,
            f"Agent {agent_name}",
            color=(i, len(agents) + 1),
            width=1.5,
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    revenue_rate_data = [[] for _ in agents]

    # Create total revenue plot
    total_revenue_chart = create_chart(
        layout, legend_width=300, x_label="Timestep", y_label="Total revenue"
    )

    total_revenue_plots = [
        add_line_plot(
            total_revenue_chart,
            f"Agent {agent_name}",
            color=(i, len(agents) + 1),
            width=1.5,
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    total_revenue_data = [[] for _ in agents]

    for i in range(args.iterations):
        logging.debug("=" * 20 + " step %s " + "=" * 20, i)

        # X. Visualize the environment.
        if i % args.fast_forward_factor == 0:
            # Plot environment.
            env_x, env_y = await environment.generate_plot_data(min_x, max_x)
            env_plot.setData(env_x, env_y)

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

                # Get data.
                data = await agent.generate_plot_data(min_x, max_x, logspace=LOG_PLOT)
                agent_x = data.pop("x")
                agent_y = data["policy"]
                agents_dist[agent_id].setData(agent_x, agent_y)

                agent_qps_plots[agent_id].setData(queries_per_second[agent_id])

                # Plot init policy and add it to last list in container.
                if "init policy" in data.keys():
                    init_agent_y = data["init policy"]
                    agents_init_dist[agent_id].setData(agent_x, init_agent_y)

                # Agent q/s.
                agent_qps_x = min(max_x, max(min_x, scaled_bids[agent_id]))
                agents_scatter_qps[agent_id].setData(
                    [agent_qps_x], [queries_per_second[agent_id][-1]]
                )

                # Total queries served by agent
                total_agent_queries_plots[agent_id].setData(
                    total_agent_queries_data[agent_id]
                )

                # Revenue rate by agent
                revenue_rate_plots[agent_id].setData(revenue_rate_data[agent_id])

                # Total revenue by agent
                total_revenue_plots[agent_id].setData(total_revenue_data[agent_id])

            # Total queries unserved
            total_unserved_queries_plot.setData(total_unserved_queries_data)

            policy_chart.setTitle(f"time {i}")

        QtWidgets.QApplication.processEvents()  # type: ignore

        if args.save:
            if i % args.fast_forward_factor == 0:
                if not ffmpeg_process:
                    # Start ffmpeg to save video
                    FILENAME = f"{args.config}.mp4"
                    ffmpeg_process = create_video_process(FILENAME, WINDOW_SIZE)

                render_video_frame(ffmpeg_process, layout, WINDOW_SIZE)

        else:  # Show
            if layout.isHidden():
                break

    if ffmpeg_process:
        close_video_process(ffmpeg_process)

    if layout.isHidden():
        pg.exit()
    else:
        # Keep window open
        pg.exec()


if __name__ == "__main__":
    run(main())
