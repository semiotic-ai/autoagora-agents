# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from asyncio import run

from environments.simulated_subgraph import SimulatedSubgraph
from simulation.chart import ChartsWidget, IndexedColor, NamedColor, PenConfig, PenStyle
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

    output_file = f"{args.config}.mp4" if args.save else None

    colors = [IndexedColor(i, len(agents) + 1) for i in range(len(agents) + 1)]

    charts = ChartsWidget(
        title="Multi-agent training",
        output_file=output_file,
        antialias=True,
        foreground="w",
    )

    policy_chart = charts.create_chart(
        title="time 0",
        height=300,
        legend_width=300,
        x_label="Price multiplier",
        y_label="Query rate",
        x_log=True,
        y_range=(0, 1.3),
    )

    query_rate_chart = charts.create_chart(
        legend_width=300, x_label="Timestep", y_label="Query rate"
    )

    total_queries_chart = charts.create_chart(
        legend_width=300, x_label="Timestep", y_label="Total queries"
    )

    # Create revenue rate plot
    revenue_rate_chart = charts.create_chart(
        legend_width=300, x_label="Timestep", y_label="Revenue rate"
    )

    # Create total revenue plot
    total_revenue_chart = charts.create_chart(
        legend_width=300, x_label="Timestep", y_label="Total revenue"
    )

    for i, agent_name in enumerate(agents.keys()):
        agent_key = f"a{i}"
        policy_chart.add_line_plot(
            agent_key, f"Agent {agent_name}: policy", color=colors[i], width=1.5
        )
        policy_chart.add_line_plot(
            f"i{i}",
            f"Agent {agent_name}: init policy",
            color=colors[i],
            style=PenStyle.DotLine,
            width=1.5,
        )
        policy_chart.add_scatter_plot(
            f"q{i}",
            f"Agent {agent_name}: query rate",
            color=colors[i],
            border=PenConfig(color=NamedColor.w),
        )
        query_rate_chart.add_line_plot(
            agent_key, f"Agent {agent_name}", color=colors[i], width=1.5
        )
        total_queries_chart.add_line_plot(
            agent_key, f"Agent {agent_name}", color=colors[i], width=1.5
        )
        revenue_rate_chart.add_line_plot(
            agent_key, f"Agent {agent_name}", color=colors[i], width=1.5
        )
        total_revenue_chart.add_line_plot(
            agent_key, f"Agent {agent_name}", color=colors[i], width=1.5
        )

    policy_chart.add_line_plot(
        "e", "Environment: total query rate", color="gray", width=1.5
    )

    total_queries_chart.add_line_plot(
        "d", "Dropped", color=colors[len(agents)], width=1.5
    )

    queries_per_second = [[] for _ in agents]
    revenue_rate_data = [[] for _ in agents]
    total_revenue_data = [[] for _ in agents]
    total_agent_queries_data = [[] for _ in agents]
    total_unserved_queries_data = []

    for i in range(args.iterations):
        if not args.save and charts.is_hidden:
            break

        logging.debug("=" * 20 + " step %s " + "=" * 20, i)

        # X. Visualize the environment.
        if i % args.fast_forward_factor == 0:
            # Plot environment.
            env_x, env_y = await environment.generate_plot_data(min_x, max_x)
            policy_chart.set_data("e", env_x, env_y)

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

                agent_key = f"a{agent_id}"

                # Get data.
                data = await agent.generate_plot_data(min_x, max_x, logspace=LOG_PLOT)
                agent_x = data.pop("x")
                agent_y = data["policy"]
                policy_chart.set_data(agent_key, agent_x, agent_y)
                query_rate_chart.set_data(agent_key, queries_per_second[agent_id])

                # Plot init policy and add it to last list in container.
                if "init policy" in data.keys():
                    init_agent_y = data["init policy"]
                    policy_chart.set_data(f"i{agent_id}", agent_x, init_agent_y)

                # Agent q/s.
                agent_qps_x = min(max_x, max(min_x, scaled_bids[agent_id]))

                policy_chart.set_data(
                    f"q{agent_id}", [agent_qps_x], [queries_per_second[agent_id][-1]]
                )

                # Total queries served by agent
                total_queries_chart.set_data(
                    agent_key, total_agent_queries_data[agent_id]
                )

                # Revenue rate by agent
                revenue_rate_chart.set_data(agent_key, revenue_rate_data[agent_id])

                # Total revenue by agent
                total_revenue_chart.set_data(agent_key, total_revenue_data[agent_id])

            # Total queries unserved
            total_queries_chart.set_data("d", total_unserved_queries_data)

            policy_chart.title = f"time {i}"

            charts.render()

    charts.close()


if __name__ == "__main__":
    run(main())
