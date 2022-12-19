# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from subprocess import Popen
from typing import Any, Dict, Literal, NamedTuple, Optional, Tuple, TypedDict, Union

import ffmpeg
import pyqtgraph as pg
from multimethod import multimethod
from pyqtgraph.Qt import QtCore, QtGui


class RGBAColor(NamedTuple):
    """Color model based on red, green, blue and an alpha/opacity value.

    Attributes:
        R (int): value between 0 and 255 for red value.
        G (int): value between 0 and 255 for green value.
        R (int): value between 0 and 255 for blue value.
        A (int, optional): value between 0 and 255 for alpha/opacity value.
    """

    R: int
    G: int
    B: int
    A: Optional[int]


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


@multimethod
def _make_color(color: RGBAColor):  # pyright:ignore [reportGeneralTypeIssues]
    """Convert RGBA color value to QColor"""
    if color.A is None:
        return pg.mkColor((color.R, color.G, color.B))
    else:
        return pg.mkColor((color.R, color.G, color.B, color.A))


@multimethod
def _make_color(color: IndexedColor):  # pyright:ignore [reportGeneralTypeIssues]
    """Convert Indexed color value to QColor"""
    return pg.mkColor((color.index, color.hues))


@multimethod
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


def _make_pen(
    color: Optional[Color] = None,
    width: Optional[float] = None,
    style: Optional[PenStyle] = None,
) -> Any:
    """Create a QPen from provided parameters"""
    config: Dict[str, Any] = {}

    if color is not None:
        config["color"] = _make_color(color)

    if width is not None:
        config["width"] = width

    if style is not None:
        config["style"] = style.value

    return pg.mkPen(**config)


SymbolType = Literal["o", "x", "s"]  # dot  # cross  # square


class PenConfig(TypedDict, total=False):
    """Configuration for pen to draw lines"""

    width: float
    """Width of the line"""
    style: PenStyle
    """Line style"""
    color: Color
    """Line color"""


# Method to resample the data before plotting ot avoid plotting multiple line segments per pixel.
DownsampleMethod = Literal[
    "subsample",  # Downsample by taking the first of N samples. This method is fastest and least accurate.
    "mean",  # Downsample by taking the mean of N samples.
    "peak",  # Downsample by drawing a saw wave that follows the min and max of the original data. This method produces the best visual representation of the data but is slower.
]

DEFAULT_WIN_SIZE = (1000, 1000)
DEFAULT_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"


class ChartWidget:
    """A chart container for single group of plots and a legend.

    Args:
        chart (PlotItem): base container plotitem
    """

    def __init__(self, chart: pg.PlotItem) -> None:
        self.__chart = chart
        self.__plots: Dict[str, pg.PlotDataItem] = {}

    @staticmethod
    def _update_downsapling(
        plot: pg.PlotDataItem,
        auto: Optional[bool] = None,
        ds: Optional[int] = None,
        method: Optional[DownsampleMethod] = None,
    ):
        """Update plot downsampling parameters"""

        if auto is None and ds is None and method is None:
            return

        config: Dict[str, Any] = {}

        if auto is not None:
            config["auto"] = auto

        if ds is not None:
            config["ds"] = ds

        if method is not None:
            config["method"] = method

        plot.setDownsampling(**config)

    def add_line_plot(
        self,
        id: str,
        name: str,
        color: Optional[Color] = None,
        width: Optional[float] = None,
        style: Optional[PenStyle] = None,
        autoDownsample: Optional[bool] = None,
        downsample: Optional[int] = None,
        downsampleMethod: Optional[DownsampleMethod] = None,
    ):
        """Add a line plot to the chart.

        Args:
            id (str): unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            color (Color, optional): Line color of the plot. Defaults to None.
            width (float, optional): Width of the plot line. Defaults to None.
            style (PenStyle, optional): Style of the plot line. Defaults to None.
            autoDownsample (bool, optional): If True, resample the data before plotting to avoid plotting multiple line segments per pixel. Defaults to None.
            downsample (int,optional): Reduce the number of samples displayed by the given factor. Defaults  to None.
            downsampleMethod (DownsampleMethod,optional): Method to use for downsampling. Defaults to None.
        """
        config = {"name": name}
        config["pen"] = _make_pen(color=color, width=width, style=style)
        plot = self.__chart.plot(**config)
        ChartWidget._update_downsapling(
            plot, auto=autoDownsample, ds=downsample, method=downsampleMethod
        )

        self.__plots[id] = plot

    def add_scatter_plot(
        self,
        id: str,
        name: str,
        size: Optional[float] = None,
        color: Optional[Color] = None,
        symbol: Optional[SymbolType] = None,
        border: Optional[PenConfig] = None,
        autoDownsample: Optional[bool] = None,
        downsample: Optional[int] = None,
        downsampleMethod: Optional[DownsampleMethod] = None,
    ):
        """Add a scatter plot to the chart.

        Args:
            id (str): Unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            size (float, optional): Size of the marker symbol. Defaults to None.
            color (Color, optional): Color to fill the marker symbol. Defaults to None.
            symbol (SymbolType, optional): Shape of the marker symbol. Defaults to None.
            border (PenConfig, optional): Pen to draw border around the marker symbol. Defaults to None.
            autoDownsample (bool, optional): If True, resample the data before plotting to avoid plotting multiple line segments per pixel. Defaults to None.
            downsample (int,optional): Reduce the number of samples displayed by the given factor. Defaults  to None.
            downsampleMethod (DownsampleMethod,optional): Method to use for downsampling. Defaults to None.
        """
        config = {"name": name, "pen": None}

        if size is not None:
            config["symbolSize"] = size

        if color is not None:
            config["symbolBrush"] = _make_color(color)

        if symbol is not None:
            config["symbol"] = symbol

        if border is not None:
            config["symbolPen"] = _make_pen(**border)

        plot = self.__chart.plot(**config)
        ChartWidget._update_downsapling(
            plot, auto=autoDownsample, ds=downsample, method=downsampleMethod
        )

        self.__plots[id] = plot

    def set_data(self, id: str, *args, **kargs):
        """Update the plot with provided data \n
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
    """A single column layout with each row containing a single chart

    Args:
        title (str): Title of charts window.
        output_file (str, optional): Path of the captured video output file. If not empty, hides UI. Defaults to None.
        output_codec (str, optional): Codec name to be used for encoding. Defaults to "libx264".
        output_pixel_format (str, optional): Ouput pixel format. Defaults to "yuv420p".
        size (Tuple[int,int]):(width,height) of the charts window. Defaults to 1000x1000.
        antialias (bool, optional): Use antialiasing. If true, smooth visuals and slower refresh rate. Defaults to None.
        foreground (Color, optional): General foreground color (text,ie). Defaults to None.
        background (Color, optional): General background color. Defaults to None.
        border (Union[bool,Tuple[int,int,int]], optional): Border between charts.`True` for default border, `False`for None or triplet of int for custom. Defaults to None.
    """

    def __init__(
        self,
        title: str,
        output_file: Optional[str] = None,
        output_codec: str = DEFAULT_CODEC,
        output_pixel_format: str = DEFAULT_PIXEL_FORMAT,
        size: Tuple[int, int] = DEFAULT_WIN_SIZE,
        antialias: Optional[bool] = None,
        foreground: Optional[Color] = None,
        background: Optional[Color] = None,
        border: Optional[Union[bool, Tuple[int, int, int]]] = None,
    ) -> None:
        self.__save = output_file is not None

        # Set up PyQtGraph
        if foreground is not None:
            pg.setConfigOption("foreground", _make_color(foreground))
        if background is not None:
            pg.setConfigOption("background", _make_color(background))
        if antialias is not None:
            pg.setConfigOptions(antialias=antialias)

        self.__app = pg.mkQApp(title)

        self.__layout = pg.GraphicsLayoutWidget(
            show=not self.__save, title=title, size=size, border=border
        )

        self.__init_size = size

        if output_file is not None:
            self.__ffmpeg_process: Popen = (
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

        chart = self.__layout.addPlot(title=title)
        if height is not None:
            chart.setPreferredHeight(300)
        legend = chart.addLegend(offset=None)
        vb: pg.ViewBox = self.__layout.addViewBox()
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

        self.__layout.nextRow()

        return ChartWidget(chart)

    def render(self):
        """Render chart changes to UI or animation file."""
        self.__app.processEvents()  # type: ignore

        if self.__save:
            # Renders a frame by capturing plots drawn on the layout.Re-scales if required based on the size.
            qimage = self.__layout.grab().toImage()

            qimage = qimage.convertToFormat(
                QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
            )

            # May have to rescale (HiDPI displays, etc)
            if (qimage.width(), qimage.height()) != self.__init_size:
                qimage = (
                    QtGui.QPixmap.fromImage(qimage)  # type: ignore
                    .scaled(
                        self.__init_size[0],
                        self.__init_size[1],
                        mode=QtCore.Qt.TransformationMode.SmoothTransformation,  # type: ignore
                    )
                    .toImage()
                    .convertToFormat(
                        QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
                    )
                )

            self.__ffmpeg_process.stdin.write(  # pyright: ignore [reportOptionalMemberAccess]
                qimage.constBits().tobytes()
            )

    @property
    def is_hidden(self) -> bool:
        """Check if UI is hidden. Always true when save file is defined."""
        return self.__layout.isHidden()

    def close(self):
        """Close the UI or finish animation file recording."""
        if self.__save:
            self.__ffmpeg_process.stdin.close()  # pyright: ignore [reportOptionalMemberAccess]
            self.__ffmpeg_process.wait()

        if self.is_hidden:
            pg.exit()
        else:
            pg.exec()  # display till user closes the UI
