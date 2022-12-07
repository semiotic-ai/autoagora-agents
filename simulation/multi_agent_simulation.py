# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from asyncio import run
from subprocess import Popen
from typing import Optional, Tuple

import ffmpeg
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from environments.simulated_subgraph import SimulatedSubgraph
from simulation.controller import init_simulation
from simulation.show_bandit import add_experiment_argparse

logging.basicConfig(level="WARN", format="%(message)s")

LOG_PLOT = True
WINDOW_SIZE = (1000, 1000)


def create_layout(
    size: Tuple[int, int],
    title: str,
    show: bool,
    antialias: Optional[bool] = None,
    foreground=None,
) -> pg.GraphicsLayoutWidget:
    """Initialize application and layout container

    Args:
        size (Tuple[int,int]):(width,height) of the charts window.
        title (str): Title of charts window.
        show (bool): Show app window.
        antialias (Optional[bool], optional): Use antialiasing. If true, smooth visuals and slower refresh rate. Defaults to None.
        foreground (Any, optional): General foreground color (text,ie). Defaults to None.
    """
    # Set up PyQtGraph
    if foreground is not None:
        pg.setConfigOption("foreground", foreground)
    if antialias is not None:
        pg.setConfigOptions(antialias=antialias)

    pg.mkQApp(title)

    layout = pg.GraphicsLayoutWidget(show=show, title=title)
    layout.resize(*size)
    return layout


def create_chart(
    layout: pg.GraphicsLayoutWidget,
    title: Optional[str] = None,
    height: Optional[int] = None,
    legend_width: Optional[int] = None,
    x_label: Optional[str] = None,
    x_range: Optional[Tuple[float, float]] = None,
    x_log: Optional[bool] = None,
    y_label: Optional[str] = None,
    y_range: Optional[Tuple[float, float]] = None,
    y_log: Optional[bool] = None,
) -> pg.PlotItem:
    """Add a chart with legend for plots as a row

    Args:
        layout(GraphicsLayoutWidget): Parent layout
        title (str, optional): Title to display on the top. Defaults to None.
        height (int, optional): Prefred height. Defaults to None.
        legend_width (int, optional):Legend width. Defaults to None.
        x_label (str, optional): Label text to display under x axis. Defaults to None.
        x_range (Tuple[float,float], optional): Constant range(min,max value) of x axis. Defaults to None.
        x_log (bool, optional): Use logarithmic scale to display x axis. Defaults to None.
        y_label (str, optional): Label text to display next to y axis. Defaults to None.
        y_range (Tuple[float,float], optional): Constant range(min,max value) of y axis. Defaults to None.
        y_log (bool, optional): Use logarithmic scale for y axis. Defaults to None.
    Returns:
        PlotItem: a chart to contain plots
    """
    chart = layout.addPlot(title=title)
    if height is not None:
        chart.setPreferredHeight(300)
    legend = chart.addLegend(offset=None)
    vb: pg.ViewBox = layout.addViewBox()
    if legend_width is not None:
        vb.setFixedWidth(legend_width)
    legend.setParentItem(vb)
    legend.anchor((0, 0), (0, 0))
    chart.setClipToView(True)

    if x_label is not None:
        chart.setLabel("bottom", x_label)
    if x_log is not None:
        chart.setLogMode(x=x_log)
    if x_range is not None:
        chart.setXRange(*x_range)

    if y_label is not None:
        chart.setLabel("left", y_label)
    if y_log is not None:
        chart.setLogMode(y=y_log)
    if y_range is not None:
        chart.setYRange(*y_range)

    layout.nextRow()

    return chart


def add_line_plot(
    chart: pg.PlotItem, name: str, color=None, width: Optional[float] = None, style=None
) -> pg.PlotDataItem:
    """Adds a line plot to the chart.

    Args:
        name (str): Name (legend title) of the plot
        color (Any, optional): Line color of the plot. Defaults to None.
        width (float, optional): Width of the plot line. Defaults to None.
        style (Any, optional): Style of the plot line. Defaults to None.
    """
    config = {}
    if color is not None:
        config["color"] = color
    if width is not None:
        config["width"] = width
    if style is not None:
        config["style"] = style

    pen = pg.mkPen(**config)
    return chart.plot(name=name, pen=pen)


def add_scatter_plot(
    chart: pg.PlotItem,
    name: str,
    size: Optional[float] = None,
    color=None,
    symbol=None,
    border=None,
) -> pg.PlotDataItem:
    """Adds a scatter plot to the chart.

    Args:
        name (str): Name (legend title) of the plot
        size (float, optional): Size of the marker symbol. Defaults to None.
        color (Any, optional): Color to fill the marker symbol. Defaults to None.
        symbol (Any, optional): Shape of the marker symbol. Defaults to None.
        border (Any, optional): Pen to draw border around the marker symbol. Defaults to None.
    """
    config = {}

    if size is not None:
        config["symbolSize"] = size

    if color is not None:
        config["symbolBrush"] = color

    if symbol is not None:
        config["symbol"] = symbol

    if border is not None:
        config["symbolPen"] = border

    return chart.plot(name=name, pen=None, **config)


def create_video_process(
    file_name: str,
    size: Tuple[int, int],
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
) -> Popen:
    """Creates a ffmpeg video process which accepts input from stdin.(NA in windows)

    Args:
        file_name (str): name of the file to save encoded video ouput
        size (Tuple[int,int]): Size (width,height) of the video frame
        codec (str, optional): Codec name to be used for encoding. Defaults to "libx264".
        pixel_format (str, optional): Ouput pixel format. Defaults to "yuv420p".

    Returns:
        Process: Asyncronious sub process listening stdin for encoding to the file.
    """
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{size[0]}x{size[1]}",
        )
        .output(file_name, vcodec=codec, pix_fmt=pixel_format)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    return process


def render_video_frame(
    video_process: Popen, layout: pg.GraphicsLayoutWidget, size: Tuple[int, int]
):
    """Renders a frame by capturing plots drawn on the layout.
    Re-scales if required based on the size.

    Args:
        video_process (Process): Video (ffmpeg) sub process
        layout (pg.GraphicsLayoutWidget): Layout to capture frame
        size (Tuple[int,int]): Output size defined while creating the video process.
    """

    qimage = layout.grab().toImage()

    qimage = qimage.convertToFormat(
        QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
    )

    # May have to rescale (HiDPI displays, etc)
    if (qimage.width(), qimage.height()) != size:
        qimage = (
            QtGui.QPixmap.fromImage(qimage)  # type: ignore
            .scaled(
                size[0],
                size[1],
                mode=QtCore.Qt.TransformationMode.SmoothTransformation,  # type: ignore
            )
            .toImage()
            .convertToFormat(
                QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
            )
        )

    video_process.stdin.write(  # pyright: ignore [reportOptionalMemberAccess]
        qimage.constBits().tobytes()
    )


def close_video_process(video_process: Popen):
    """Waits until encoding sub process is finished data in the stdin

    Args:
        video_process (Process): Vide (ffmpeg) sub process
    """
    video_process.stdin.close()  # pyright: ignore [reportOptionalMemberAccess]
    video_process.wait()


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
