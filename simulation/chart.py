# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from subprocess import Popen
from typing import Any, Dict, NamedTuple, Optional, Tuple, TypedDict

import ffmpeg
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class RGBAColor(NamedTuple):
    """An RGB color value represents RED, GREEN, and BLUE light sources
    with alpha channel- which specifies the opacity for a color."""

    R: int
    """Red [0-255]"""
    G: int
    """Green [0-255]"""
    B: int
    """Blue [0-255]"""
    A: Optional[int]
    """Alpha [0-255]"""


class NamedColor(Enum):
    "Single-character string representing a predefined color"
    r = "r"
    """Red"""
    g = "g"
    """Green"""
    b = "b"
    """Blue"""
    c = "c"
    """Cyan"""
    m = "m"
    """Magenta"""
    y = "ky"
    """Yellow"""
    k = "k"
    """Key (black)"""
    w = "w"
    """White"""


class IndexedColor(NamedTuple):
    """Color from a single index. Useful for stepping through a predefined list of colors."""

    index: int
    hues: int


Color = RGBAColor | NamedColor | IndexedColor | int | float | str
"""Color based on model, predefined values, single greyscale value (0.0 - 1.0) , color index or string representing color (“#RGB”,“#RGBA”,“#RRGGBB”,“#RRGGBBAA”)."""


def _make_color(color: Color) -> Any:
    """Converts color value to QColor"""
    if isinstance(color, RGBAColor):
        if color.A is None:
            return pg.mkColor((color.R, color.G, color.B))
        else:
            return pg.mkColor((color.R, color.G, color.B, color.A))
    elif isinstance(color, NamedColor):
        return pg.mkColor(color.value)
    elif isinstance(color, IndexedColor):
        return pg.mkColor((color.index, color.hues))
    else:
        return pg.mkColor(color)


class PenStyle(Enum):
    """Pen style for drawing lines"""

    # NoPen                    = QtCore.Qt.PenStyle.NoPen
    SolidLine = QtCore.Qt.PenStyle.SolidLine  # pyright:ignore [reportGeneralTypeIssues]
    DashLine = QtCore.Qt.PenStyle.DashLine  # pyright:ignore [reportGeneralTypeIssues]
    DotLine = QtCore.Qt.PenStyle.DotLine  # pyright:ignore [reportGeneralTypeIssues]
    DashDotLine = (
        QtCore.Qt.PenStyle.DashDotLine  # pyright:ignore [reportGeneralTypeIssues]
    )
    DashDotDotLine = (
        QtCore.Qt.PenStyle.DashDotDotLine  # pyright:ignore [reportGeneralTypeIssues]
    )
    # CustomDashLine           = QtCore.Qt.PenStyle.CustomDashLine
    # MPenStyle                = QtCore.Qt.PenStyle.MPenStyle


def _make_pen(
    color: Optional[Color] = None,
    width: Optional[float] = None,
    style: Optional[PenStyle] = None,
) -> Any:
    """Creates a QPen from a parameters"""
    config: Dict[str, Any] = {}

    if color is not None:
        config["color"] = _make_color(color)

    if width is not None:
        config["width"] = width

    if style is not None:
        config["style"] = style.value

    return pg.mkPen(**config)


class SymbolType(Enum):
    """Marker symbol shape type"""

    dot = "o"
    """Circle"""
    cross = "x"
    """Cross"""
    square = "s"
    """Square"""


class PenConfig(TypedDict, total=False):
    """Configuration for pen to draw lines"""

    width: float
    """Width of the line"""
    style: PenStyle
    """Line style"""
    color: Color
    """Line color"""


def _create_chart(
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


def _add_line_plot(
    chart: pg.PlotItem,
    name: str,
    color: Optional[Color] = None,
    width: Optional[float] = None,
    style: Optional[PenStyle] = None,
) -> pg.PlotDataItem:
    """Adds a line plot to the chart.

    Args:
        name (str): Name (legend title) of the plot
        color (Color, optional): Line color of the plot. Defaults to None.
        width (float, optional): Width of the plot line. Defaults to None.
        style (PenStyle, optional): Style of the plot line. Defaults to None.
    """

    pen = _make_pen(color=color, width=width, style=style)
    return chart.plot(name=name, pen=pen)


def _add_scatter_plot(
    chart: pg.PlotItem,
    name: str,
    size: Optional[float] = None,
    color: Optional[Color] = None,
    symbol: Optional[SymbolType] = None,
    border: Optional[PenConfig] = None,
) -> pg.PlotDataItem:
    """Adds a scatter plot to the chart.

    Args:
        name (str): Name (legend title) of the plot
        size (float, optional): Size of the marker symbol. Defaults to None.
        color (Color, optional): Color to fill the marker symbol. Defaults to None.
        symbol (SymbolType, optional): Shape of the marker symbol. Defaults to None.
        border (PenConfig, optional): Pen to draw border around the marker symbol. Defaults to None.
    """
    config = {}

    if size is not None:
        config["symbolSize"] = size

    if color is not None:
        config["symbolBrush"] = _make_color(color)

    if symbol is not None:
        config["symbol"] = symbol.value

    if border is not None:
        config["symbolPen"] = _make_pen(**border)

    return chart.plot(name=name, pen=None, **config)


def _create_video_process(
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


def _render_video_frame(
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


def _close_video_process(video_process: Popen):
    """Waits until encoding sub process is finished data in the stdin

    Args:
        video_process (Process): Vide (ffmpeg) sub process
    """
    video_process.stdin.close()  # pyright: ignore [reportOptionalMemberAccess]
    video_process.wait()


DEFAULT_WIN_SIZE = (1000, 1000)
DEFAULT_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"


class ChartWidget:
    def __init__(self, chart: pg.PlotItem) -> None:
        self.__chart = chart
        self.__plots: Dict[str, pg.PlotDataItem] = {}

    def add_line_plot(
        self,
        id: str,
        name: str,
        color: Optional[Color] = None,
        width: Optional[float] = None,
        style: Optional[PenStyle] = None,
    ):
        """Adds a line plot to the chart.

        Args:
            id (str): unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            color (Color, optional): Line color of the plot. Defaults to None.
            width (float, optional): Width of the plot line. Defaults to None.
            style (PenStyle, optional): Style of the plot line. Defaults to None.
        """
        self.__plots[id] = _add_line_plot(
            chart=self.__chart, name=name, color=color, width=width, style=style
        )

    def add_scatter_plot(
        self,
        id: str,
        name: str,
        size: Optional[float] = None,
        color: Optional[Color] = None,
        symbol: Optional[SymbolType] = None,
        border: Optional[PenConfig] = None,
    ):
        """Adds a scatter plot to the chart.

        Args:
            id (str): Unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            size (float, optional): Size of the marker symbol. Defaults to None.
            color (Color, optional): Color to fill the marker symbol. Defaults to None.
            symbol (SymbolType, optional): Shape of the marker symbol. Defaults to None.
            border (PenConfig, optional): Pen to draw border around the marker symbol. Defaults to None.
        """
        self.__plots[id] = _add_scatter_plot(
            chart=self.__chart,
            name=name,
            size=size,
            color=color,
            symbol=symbol,
            border=border,
        )

    def set_data(self, id: str, *args, **kargs):
        """Updates the plot with provided data \n
        set_data(id, x, y):x, y: array_like coordinate values
        set_data(id, y): y values only – x will be automatically set to range(len(y))
        set_data(id, x=x, y=y): x and y given by keyword arguments
        set_data(id, ndarray(N,2)): single numpy array with shape (N, 2), where x=data[:,0] and y=data[:,1]

        Args:
            id (str): Unique id of the plot in the chart
        """
        self.__plots[id].setData(*args, **kargs)

    @property
    def title(self) -> str:
        """Title of the chart (get/set)"""
        return self.__chart.titleLabel.text

    @title.setter
    def title(self, val: str):
        self.__chart.setTitle(title=val)


class ChartsWidget:
    def __init__(
        self,
        title: str,
        output_file: Optional[str] = None,
        output_codec: str = DEFAULT_CODEC,
        output_pixel_format: str = DEFAULT_PIXEL_FORMAT,
        size: Tuple[int, int] = DEFAULT_WIN_SIZE,
        antialias: Optional[bool] = None,
        foreground=None,
    ) -> None:
        """Initialize application and layout container

        Args:
            title (str): Title of charts window.
            output_file (str, optional): Path of the captured video output file. If not empty, hides UI. Defaults to None.
            output_codec (str, optional): Codec name to be used for encoding. Defaults to "libx264".
            output_pixel_format (str, optional): Ouput pixel format. Defaults to "yuv420p".
            size (Tuple[int,int]):(width,height) of the charts window. Defaults to 1000x1000.
            antialias (Optional[bool], optional): Use antialiasing. If true, smooth visuals and slower refresh rate. Defaults to None.
            foreground (Any, optional): General foreground color (text,ie). Defaults to None.
        """

        self.__save = output_file is not None

        # Set up PyQtGraph
        if foreground is not None:
            pg.setConfigOption("foreground", foreground)
        if antialias is not None:
            pg.setConfigOptions(antialias=antialias)

        self.__app = pg.mkQApp(title)

        self.__layout = pg.GraphicsLayoutWidget(
            show=not self.__save, title=title, size=size
        )

        self.__init_size = size

        if output_file is not None:
            self.__ffmpeg_process = _create_video_process(
                output_file, size, codec=output_codec, pixel_format=output_pixel_format
            )

    def create_chart(
        self,
        title: Optional[str] = None,
        height: Optional[int] = None,
        legend_width: Optional[int] = None,
        x_label: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
        x_log: Optional[bool] = None,
        y_label: Optional[str] = None,
        y_range: Optional[Tuple[float, float]] = None,
        y_log: Optional[bool] = None,
    ) -> ChartWidget:
        """Add a chart with legend for plots as a row

        Args:
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
            ChartWidget: a chart to define and draw plots
        """

        chart = _create_chart(
            self.__layout,
            title=title,
            height=height,
            legend_width=legend_width,
            x_label=x_label,
            x_range=x_range,
            x_log=x_log,
            y_label=y_label,
            y_range=y_range,
            y_log=y_log,
        )
        return ChartWidget(chart)

    def render(self):
        """Renders chart changes to UI or animation file."""

        self.__app.processEvents()  # type: ignore

        if self.__save:
            _render_video_frame(self.__ffmpeg_process, self.__layout, self.__init_size)

    @property
    def is_hidden(self) -> bool:
        """Checks if UI is hidden. Always true when save file is defined."""
        return self.__layout.isHidden()

    def close(self):
        """Closes the UI or finishes animation file recording."""
        if self.__save:
            _close_video_process(self.__ffmpeg_process)

        if self.is_hidden:
            pg.exit()
        else:
            pg.exec()  # display till user closes the UI
