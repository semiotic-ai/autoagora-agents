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
        A (int): value between 0 and 255 for alpha/opacity value. Default is `255`.
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
    """Color from a single index. Useful for stepping through a predefined colors.

    Attributes:
        index (int): predefined color index
        hues (int): number of steps to generate
    """

    index: int
    hues: int


Color = Union[RGBAColor, NamedColor, IndexedColor, int, float, str]
"""Color based on provided model.

RGBAColor: RGBA structure value. 
NamedColor : predefined color
IndexedColor: single predefined color index with hue.
int: predefined color index
float (0.0 - 1.0): greyscale value
str (“#RGB”,“#RGBA”,“#RRGGBB”,“#RRGGBBAA”): web hex color encoding
"""


@singledispatch
def _make_color(color: RGBAColor):  # pyright: ignore [reportGeneralTypeIssues]
    """Convert RGBA color value to QColor"""
    return pg.mkColor((color.R, color.G, color.B, color.A))


@singledispatch
def _make_color(color: IndexedColor):  # pyright: ignore [reportGeneralTypeIssues]
    """Convert Indexed color value to QColor"""
    return pg.mkColor((color.index, color.hues))


@singledispatch
def _make_color(
    color: Union[int, float, str]
):  # pyright: ignore [reportGeneralTypeIssues]
    """Convert Indexed color value to QColor"""
    return pg.mkColor(color)


class PenStyle(Enum):
    """Pen style for drawing lines


    Attributes:
        SolidLine: simple solid line
        DashLine: all dashed line
        DotLine: all dotted line
        DashDotLine: dash dot pattern line
        DashDotDotLine: dash dot dot pattern line
    """

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
    color: Color = RGBAColor(200, 200, 200),
    width: float = 1.0,
    style: PenStyle = PenStyle.SolidLine,
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


DownsampleMethod = Literal["subsample", "mean", "peak"]
"""Method to resample the data before plotting to avoid plotting multiple line 
segments per pixel. 

`subsample`: Downsample by taking the first of N samples. This method is fastest and 
least accurate.
`mean`: Downsample by taking the mean of N samples.
`peak`: Downsample by drawing a saw wave that follows the min and max of the original 
data. This method produces the best visual representation of the data but is slower.
"""


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
        color: Color = RGBAColor(200, 200, 200),
        width: float = 1.0,
        style: PenStyle = PenStyle.SolidLine,
        auto_downsample: bool = False,
        downsample: int = 1,
        downsample_method: DownsampleMethod = "peak",
    ):
        """Add a line plot to the chart.

        Args:
            id (str): unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            color (Color): Line color of the plot. Defaults to `RGBAColor(200,200,200)`.
            width (float): Width of the plot line. Defaults to `1.0`.
            style (PenStyle): Style of the plot line. Defaults to `Penstyle.SolidLine`.
            auto_downsample (bool): If True, resample the data before plotting to avoid
                plotting multiple line segments per pixel. Defaults to `False`.
            downsample (int): Reduce the number of samples displayed by the given factor.
                To disable, set to `1`. Defaults  to `1`.
            downsample_method (DownsampleMethod): Method to use for downsampling.
                Defaults to `peak`.
        """
        config = {"name": name}
        config["pen"] = _make_pen(color=color, width=width, style=style)
        plot = self._chart.plot(**config)
        plot.setDownsampling(
            ds=downsample, auto=auto_downsample, method=downsample_method
        )
        self._plots[id] = plot

    def add_scatter_plot(
        self,
        id: str,
        name: str,
        size: float = 10.0,
        color: Color = RGBAColor(50, 50, 150),
        symbol: SymbolType = "o",
        border: PenConfig = PenConfig(color=RGBAColor(200, 200, 200)),
        auto_downsample: bool = False,
        downsample: int = 1,
        downsample_method: DownsampleMethod = "peak",
    ):
        """Add a scatter plot to the chart.

        Args:
            id (str): Unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            size (float): Size of the marker symbol. Defaults to `10.0`.
            color (Color): Color to fill the marker symbol. Defaults to
                `RGBAColor(50,50,150)`.
            symbol (SymbolType): Shape of the marker symbol. Defaults to `o`.
            border (PenConfig): Pen to draw border around the marker symbol. Defaults to
                `PenConfig(color=RGBAColor(200,200,200))`.
            auto_downsample (bool): If True, resample the data before plotting to avoid
                plotting multiple line segments per pixel. Defaults to `False`.
            downsample (int): Reduce the number of samples displayed by the given
                factor.To disable, set to `1`. Defaults  to `1`.
            downsample_method (DownsampleMethod): Method to use for downsampling.
                Defaults to `peak`.
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
            ds=downsample, auto=auto_downsample, method=downsample_method
        )
        self._plots[id] = plot

    def set_data(self, id: str, *args, **kwargs):
        """Update the plot with provided data

        set_data(id, x, y):x, y: array_like coordinate values
        set_data(id, y): y values only - x will be automatically set to range(len(y))
        set_data(id, x=x, y=y): x and y given by keyword arguments
        set_data(id, ndarray(N,2)): single numpy array with shape (N, 2), where
            x=data[:,0] and y=data[:,1]

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
        output_file (str, optional): Path of the captured video output file. If not
            `None`, hides UI. Defaults to `None`.
        output_codec (str): Codec name to be used for encoding. Defaults to `libx264`.
        output_pixel_format (str): Ouput pixel format. Defaults to `yuv420p`.
        size (Tuple[int,int]):(width,height) of the charts window. Defaults to
            `1000x1000`.
        antialias (bool): Use antialiasing. If true, smooth visuals and slower refresh
            rate. Defaults to `False`.
        foreground (Color): General foreground color (text,ie). Defaults to `d`.
        background (Color): General background color. Defaults to `k`.
        border (Union[bool,Tuple[int,int,int]], optional): Border between charts.`True`
            for default border, `False`for None or triplet of int for custom. Defaults to
            `None`.
    """

    def __init__(
        self,
        title: str,
        output_file: Optional[str] = None,
        output_codec: str = "libx264",
        output_pixel_format: str = "yuv420p",
        size: Tuple[int, int] = (1000, 1000),
        antialias: bool = False,
        foreground: Color = "d",
        background: Color = "k",
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
        height: int = 200,
        legend_width: int = 300,
        x_label: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
        x_log: bool = False,
        y_label: Optional[str] = None,
        y_range: Optional[Tuple[float, float]] = None,
        y_log: bool = False,
    ) -> ChartWidget:
        """Add a chart with legend for plots as a row

        Args:
            title (str, optional): Title to display on the top. Defaults to `None`.
            height (int): Prefered height scale factor.Defaults to `200`.
            legend_width (int): Legend width. Defaults to `300`.
            x_label (str, optional): Label text to display under x axis. Defaults to
                `None`.
            x_range (Tuple[float,float], optional): Constant range(min,max value) of x
                axis. For dynamic range, set to `None`. Defaults to `None`.
            x_log (bool): Use logarithmic scale to display x axis. Defaults to `False`.
            y_label (str, optional): Label text to display next to y axis. Defaults to
                `None`.
            y_range (Tuple[float,float], optional): Constant range(min,max value) of y
                axis. For dynamic range, set to `None`. Defaults to `None`.
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

        x_label and chart.setLabel(
            "bottom", x_label
        )  # pyright: ignore [reportUnusedExpression]
        x_range and chart.setXRange(
            *x_range
        )  # pyright: ignore [reportUnusedExpression]

        y_label and chart.setLabel(
            "left", y_label
        )  # pyright: ignore [reportUnusedExpression]
        y_range and chart.setYRange(
            *y_range
        )  # pyright: ignore [reportUnusedExpression]

        self._layout.nextRow()

        return ChartWidget(chart)

    def render(self):
        """Render chart changes to UI or animation file."""
        self._app.processEvents()  # type: ignore

        if self._save:
            # Renders a frame by capturing plots drawn on the layout.Re-scales if
            # required based on the size.
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
