# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from functools import singledispatch
from subprocess import Popen
from typing import Dict, Literal, NamedTuple, Optional, Tuple, TypedDict, Union

import ffmpeg
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class RGBAColor(NamedTuple):
    """Color model based on red, green, blue and an alpha/opacity value.

    Attributes:
        R (int): value between 0 and 255 for red value.
        G (int): value between 0 and 255 for green value.
        R (int): value between 0 and 255 for blue value.
        A (int): value between 0 and 255 for alpha/opacity value.Default is `255`
    """

    R: int
    G: int
    B: int
    A: int = 255


# Single-character string representing a predefined color
NamedColor = Literal[
    "r",  # red
    "g",  # green
    "b",  # blue
    "c",  # cyan
    "m",  # magenta
    "y",  # yellow
    "k",  # key (black)
    "w",  # white
]


class IndexedColor(NamedTuple):
    """Color from a single index. Useful for stepping through a predefined list of colors.

    Attributes:
        index (int): predefined color index
        hues (int): number of steps to generate
    """

    index: int
    hues: int


Color = Union[RGBAColor, NamedColor, IndexedColor, int, float, str]
"""Color based on model, predefined values, single greyscale value (0.0 - 1.0) , color index or string representing color (“#RGB”,“#RGBA”,“#RRGGBB”,“#RRGGBBAA”)."""


@singledispatch
def _make_color(color: RGBAColor):  # pyright:ignore [reportGeneralTypeIssues]
    """Convert RGBA color value to QColor"""
    return pg.mkColor((color.R, color.G, color.B, color.A))


@singledispatch
def _make_color(color: IndexedColor):  # pyright:ignore [reportGeneralTypeIssues]
    """Convert Indexed color value to QColor"""
    return pg.mkColor((color.index, color.hues))


@singledispatch
def _make_color(color: Union[int, float, str]):
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


# Pen defaults
DEFAULT_PEN_COLOR = RGBAColor(200, 200, 200)
DEFAULT_PEN_WIDTH = 1.0
DEFAULT_PEN_STYLE = PenStyle.SolidLine


def _make_pen(
    color: Color = DEFAULT_PEN_COLOR,
    width: float = DEFAULT_PEN_WIDTH,
    style: PenStyle = DEFAULT_PEN_STYLE,
):
    """Create a QPen from provided parameters"""
    config = {
        "color": _make_color(color),
        "width": width,
        "style": style.value,
    }
    return pg.mkPen(**config)


SymbolType = Literal["o", "x", "s"]  # dot  # cross  # square


class PenConfig(TypedDict, total=False):
    """Configuration for pen to draw lines.

    Attributes:
        width (float): width of the line
        style (PenStyle): line style
        color (Color): line color
    """

    width: float
    style: PenStyle
    color: Color


# Method to resample the data before plotting ot avoid plotting multiple line segments per pixel.
DownsampleMethod = Literal[
    "subsample",  # Downsample by taking the first of N samples. This method is fastest and least accurate.
    "mean",  # Downsample by taking the mean of N samples.
    "peak",  # Downsample by drawing a saw wave that follows the min and max of the original data. This method produces the best visual representation of the data but is slower.
]

# Layout defaults
DEFAULT_WIN_SIZE = (1000, 1000)
DEFAULT_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_FOREGROUND: Color = "d"
DEFAULT_BACKGROUND: Color = "k"
DEFAULT_ANTIALIAS = False

# Chart defaults
DEFAULT_CHART_HEIGHT = 200
DEFAULT_LEGEND_WIDTH = 300
DEFAULT_AXIS_LOG = False


# Plot defaults

DEFAULT_SYMBOL_SIZE = 10.0
DEFAULT_SYMBOL_MARKER: SymbolType = "o"
DEFAULT_SYMBOL_PEN = PenConfig(color=DEFAULT_PEN_COLOR)
DEFAULT_SYMBOL_COLOR = RGBAColor(50, 50, 150)

DEFAULT_DOWNSAMPLE_METHOD = "peak"
DEFAULT_DOWNSAMPLE_AMOUNT = 1
DEFAULT_DOWNSAMMPLE_AUTO = False


class ChartWidget:
    """A chart container for single group of plots and a legend.

    Args:
        chart (PlotItem): base container plotitem
    """

    def __init__(self, chart: pg.PlotItem) -> None:
        self._chart = chart
        self._plots: Dict[str, pg.PlotDataItem] = {}

    def add_line_plot(
        self,
        id: str,
        name: str,
        color: Color = DEFAULT_PEN_COLOR,
        width: float = DEFAULT_PEN_WIDTH,
        style: PenStyle = DEFAULT_PEN_STYLE,
        autoDownsample: bool = DEFAULT_DOWNSAMMPLE_AUTO,
        downsample: int = DEFAULT_DOWNSAMPLE_AMOUNT,
        downsampleMethod: DownsampleMethod = DEFAULT_DOWNSAMPLE_METHOD,
    ):
        """Add a line plot to the chart.

        Args:
            id (str): unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            color (Color): Line color of the plot. Defaults to `RGBAColor(200,200,200)`.
            width (float): Width of the plot line. Defaults to `1.0`.
            style (PenStyle): Style of the plot line. Defaults to `Penstyle.SolidLine`.
            autoDownsample (bool): If True, resample the data before plotting to avoid plotting multiple line segments per pixel. Defaults to `False`.
            downsample (int): Reduce the number of samples displayed by the given factor.To disable, set to `1`. Defaults  to `1`.
            downsampleMethod (DownsampleMethod): Method to use for downsampling. Defaults to `peak`.
        """
        config = {"name": name}
        config["pen"] = _make_pen(color=color, width=width, style=style)
        plot = self._chart.plot(**config)
        plot.setDownsampling(
            ds=downsample, auto=autoDownsample, method=downsampleMethod
        )
        self._plots[id] = plot

    def add_scatter_plot(
        self,
        id: str,
        name: str,
        size: float = DEFAULT_SYMBOL_SIZE,
        color: Color = DEFAULT_SYMBOL_COLOR,
        symbol: SymbolType = DEFAULT_SYMBOL_MARKER,
        border: PenConfig = DEFAULT_SYMBOL_PEN,
        autoDownsample: bool = DEFAULT_DOWNSAMMPLE_AUTO,
        downsample: int = DEFAULT_DOWNSAMPLE_AMOUNT,
        downsampleMethod: DownsampleMethod = DEFAULT_DOWNSAMPLE_METHOD,
    ):
        """Add a scatter plot to the chart.

        Args:
            id (str): Unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            size (float): Size of the marker symbol. Defaults to `10.0`.
            color (Color): Color to fill the marker symbol. Defaults to `RGBAColor(50,50,150)`.
            symbol (SymbolType): Shape of the marker symbol. Defaults to `o`.
            border (PenConfig): Pen to draw border around the marker symbol. Defaults to `PenConfig(color=RGBAColor(200,200,200))`.
            autoDownsample (bool): If True, resample the data before plotting to avoid plotting multiple line segments per pixel. Defaults to `False`.
            downsample (int): Reduce the number of samples displayed by the given factor.To disable, set to `1`. Defaults  to `1`.
            downsampleMethod (DownsampleMethod): Method to use for downsampling. Defaults to `peak`.
        """
        config = {
            "name": name,
            "pen": None,
            "symbolSize": size,
            "symbolBrush": _make_color(color),
            "symbol": symbol,
            "symbolPen": _make_pen(**border),
        }

        plot = self._chart.plot(**config)
        plot.setDownsampling(
            ds=downsample, auto=autoDownsample, method=downsampleMethod
        )
        self._plots[id] = plot

    def set_data(self, id: str, *args, **kwargs):
        """Update the plot with provided data \n
        set_data(id, x, y):x, y: array_like coordinate values
        set_data(id, y): y values only - x will be automatically set to range(len(y))
        set_data(id, x=x, y=y): x and y given by keyword arguments
        set_data(id, ndarray(N,2)): single numpy array with shape (N, 2), where x=data[:,0] and y=data[:,1]

        Args:
            id (str): Unique id of the plot in the chart
        """
        self._plots[id].setData(*args, **kwargs)

    @property
    def title(self) -> str:
        """Title of the chart (get/set)"""
        return self._chart.titleLabel.text

    @title.setter
    def title(self, val: str):
        self._chart.setTitle(title=val)


class ChartsWidget:
    """A single column layout with each row containing a single chart

    Args:
        title (str): Title of charts window.
        output_file (str, optional): Path of the captured video output file. If not `None`, hides UI. Defaults to `None`.
        output_codec (str): Codec name to be used for encoding. Defaults to `libx264`.
        output_pixel_format (str): Ouput pixel format. Defaults to `yuv420p`.
        size (Tuple[int,int]):(width,height) of the charts window. Defaults to `1000x1000`.
        antialias (bool): Use antialiasing. If true, smooth visuals and slower refresh rate. Defaults to `False`.
        foreground (Color): General foreground color (text,ie). Defaults to `d`.
        background (Color): General background color. Defaults to `k`.
        border (Union[bool,Tuple[int,int,int]], optional): Border between charts.`True` for default border, `False`for None or triplet of int for custom. Defaults to `None`.
    """

    def __init__(
        self,
        title: str,
        output_file: Optional[str] = None,
        output_codec: str = DEFAULT_CODEC,
        output_pixel_format: str = DEFAULT_PIXEL_FORMAT,
        size: Tuple[int, int] = DEFAULT_WIN_SIZE,
        antialias: bool = DEFAULT_ANTIALIAS,
        foreground: Color = DEFAULT_FOREGROUND,
        background: Color = DEFAULT_BACKGROUND,
        border: Union[bool, Tuple[int, int, int], None] = None,
    ) -> None:
        self._save = output_file is not None

        # Set up PyQtGraph
        pg.setConfigOption("foreground", _make_color(foreground))
        pg.setConfigOption("background", _make_color(background))
        pg.setConfigOptions(antialias=antialias)

        self._app = pg.mkQApp(title)

        self._layout = pg.GraphicsLayoutWidget(
            show=not self._save, title=title, size=size, border=border
        )

        self._init_size = size

        if output_file is not None:
            self._ffmpeg_process: Popen = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{size[0]}x{size[1]}",
                )
                .output(output_file, vcodec=output_codec, pix_fmt=output_pixel_format)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

    def create_chart(
        self,
        title: Optional[str] = None,
        height: int = DEFAULT_CHART_HEIGHT,
        legend_width: int = DEFAULT_LEGEND_WIDTH,
        x_label: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
        x_log: bool = DEFAULT_AXIS_LOG,
        y_label: Optional[str] = None,
        y_range: Optional[Tuple[float, float]] = None,
        y_log: bool = DEFAULT_AXIS_LOG,
    ) -> ChartWidget:
        """Add a chart with legend for plots as a row

        Args:
            title (str, optional): Title to display on the top. Defaults to None.
            height (int): Prefered height scale factor.Defaults to `200`.
            legend_width (int): Legend width. Defaults to `300`.
            x_label (str, optional): Label text to display under x axis. Defaults to `None`.
            x_range (Tuple[float,float], optional): Constant range(min,max value) of x axis. For dynamic range, set to `None`. Defaults to `None`.
            x_log (bool): Use logarithmic scale to display x axis. Defaults to `False`.
            y_label (str, optional): Label text to display next to y axis. Defaults to `None`.
            y_range (Tuple[float,float], optional): Constant range(min,max value) of y axis. For dynamic range, set to `None`. Defaults to `None`.
            y_log (bool): Use logarithmic scale for y axis. Defaults to `False`.

        Returns:
            ChartWidget: a chart to define and draw plots
        """

        chart = self._layout.addPlot(title=title)
        chart.setPreferredHeight(height)
        legend = chart.addLegend(offset=None)
        vb: pg.ViewBox = self._layout.addViewBox()
        vb.setFixedWidth(legend_width)
        legend.setParentItem(vb)
        legend.anchor((0, 0), (0, 0))
        chart.setClipToView(True)
        chart.setLogMode(x=x_log, y=y_log)

        if x_label is not None:
            chart.setLabel("bottom", x_label)
        if x_range is not None:
            chart.setXRange(*x_range)

        if y_label is not None:
            chart.setLabel("left", y_label)
        if y_range is not None:
            chart.setYRange(*y_range)

        self._layout.nextRow()

        return ChartWidget(chart)

    def render(self):
        """Render chart changes to UI or animation file."""
        self._app.processEvents()  # type: ignore

        if self._save:
            # Renders a frame by capturing plots drawn on the layout.Re-scales if required based on the size.
            qimage = self._layout.grab().toImage()

            qimage = qimage.convertToFormat(
                QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
            )

            # May have to rescale (HiDPI displays, etc)
            if (qimage.width(), qimage.height()) != self._init_size:
                qimage = (
                    QtGui.QPixmap.fromImage(qimage)  # type: ignore
                    .scaled(
                        self._init_size[0],
                        self._init_size[1],
                        mode=QtCore.Qt.TransformationMode.SmoothTransformation,  # type: ignore
                    )
                    .toImage()
                    .convertToFormat(
                        QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
                    )
                )

            self._ffmpeg_process.stdin.write(  # pyright: ignore [reportOptionalMemberAccess]
                qimage.constBits().tobytes()
            )

    @property
    def is_hidden(self) -> bool:
        """Check if UI is hidden. Always true when save file is defined."""
        return self._layout.isHidden()

    def close(self):
        """Close the UI or finish animation file recording."""
        if self._save:
            self._ffmpeg_process.stdin.close()  # pyright: ignore [reportOptionalMemberAccess]
            self._ffmpeg_process.wait()

        if self.is_hidden:
            pg.exit()
        else:
            pg.exec()  # display till user closes the UI
