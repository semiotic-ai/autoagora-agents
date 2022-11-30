# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0
import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict

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
    "Single-character string representing colors"
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


class PenStyle(Enum):
    """Pen style for drawing lines"""

    # NoPen                    = QtCore.Qt.PenStyle.NoPen
    SolidLine = QtCore.Qt.PenStyle.SolidLine
    DashLine = QtCore.Qt.PenStyle.DashLine
    DotLine = QtCore.Qt.PenStyle.DotLine
    DashDotLine = QtCore.Qt.PenStyle.DashDotLine
    DashDotDotLine = QtCore.Qt.PenStyle.DashDotDotLine
    # CustomDashLine           = QtCore.Qt.PenStyle.CustomDashLine
    # MPenStyle                = QtCore.Qt.PenStyle.MPenStyle


class SymbolType(Enum):
    """Marker symbol shape type"""

    dot = "o"
    """Circle"""
    cross = "x"
    """Cross"""
    square = "s"
    """Square"""


class DownSampleMethod(Enum):
    """Method to resample the data before plotting ot avoid plotting multiple line segments per pixel."""

    subsample = "subsample"
    """Downsample by taking the first of N samples. This method is fastest and least accurate. """
    mean = "mean"
    """Downsample by taking the mean of N samples. """
    peak = "peak"
    """Downsample by drawing a saw wave that follows the min and max of the original data. 
    This method produces the best visual representation of the data but is slower."""


class PenConfig(TypedDict, total=False):
    """Configuration for pen to draw lines"""

    width: float
    """Width of the line"""
    style: PenStyle
    """Line style"""
    color: Color
    """Line color"""


class SymbolConfig(TypedDict, total=False):
    """Base config for markers"""

    symbolBrush: Color
    """Color to fill the marker"""
    symbolSize: float
    """Size of the symbol"""
    symbolPen: PenConfig
    """Border line pen"""
    symbol: SymbolType
    """Symbol shape type"""


class PlotConfig(SymbolConfig, total=False):
    """Configuration of a plot in a chart"""

    autoDownsample: bool
    """Use auto downsampling"""
    downsample: int
    """Reduce the number of samples displayed by the given factor."""
    downsampleMethod: DownSampleMethod
    """Downsample method"""
    pen: Optional[PenConfig]
    """Pen to draw line of a plot"""


class AxisConfig(TypedDict, total=False):
    """Axis configuration (x,y)"""

    label: str
    """Label text to show next to axis"""
    range: Tuple[float, float]
    """Fixed range(min,max) for axis"""
    log: bool
    """Use logarithmic scaling for axis"""


class ChartConfig(PlotConfig, total=False):
    """Chart configuration"""

    chart_height: int
    """Prefered height for the chart."""
    legend_width: int
    """Width of the legend on the right."""
    x_axis: AxisConfig
    """Configuration of x axis (horizontal)"""
    y_axis: AxisConfig
    """Configuration of y axis (vertical)"""


DEFAULT_PEN: PenConfig = {
    "width": 1,
    "style": PenStyle.SolidLine,
    "color": "gray",
}

DEFAULT_SYMBOL: PlotConfig = {
    "symbol": SymbolType.dot,
    "symbolBrush": "w",
    "symbolSize": 10,
    "symbolPen": DEFAULT_PEN,
}

DEFAULT_CHART: ChartConfig = {
    "chart_height": 300,
    "legend_width": 300,
    "x_axis": {},
    "y_axis": {},
    "pen": DEFAULT_PEN,
}

DEFAULT_WINDOW_SIZE = (1000, 1000)


class ChartWidget(ABC):
    """Proxy class for a chart"""

    @abstractmethod
    def add_line(
        self,
        id: str,
        name: str,
        color: Color = None,
        width: float = None,
        style: PenStyle = None,
    ):
        """Adds a line plot to the chart.

        Args:
            id (str): Unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            color (Color, optional): Line color of the plot. Defaults to None. Inherits charts default plot configuration.
            width (float, optional): Width of the plot line. Defaults to None. Inherits charts default plot configuration.
            style (PenStyle, optional): Style of the plot line. Defaults to None. Inherits charts default plot configuration.
        """
        ...

    @abstractmethod
    def add_scatter(
        self,
        id: str,
        name: str,
        size: float = None,
        color: Color = None,
        symbol: SymbolType = None,
        border: PenConfig = None,
    ):
        """Adds a scatterplot to the chart.

        Args:
            id (str): Unique id of the plot in the chart
            name (str): Name (legend title) of the plot
            size (float, optional): Size of the marker symbol. Defaults to None. Inherits charts default plot configuration.
            color (Color, optional): Color to fill the marker symbol. Defaults to None. Inherits charts default plot configuration.
            symbol (SymbolType, optional): Shape of the marker symbol. Defaults to None. Inherits charts default plot configuration.
            border (PenConfig, optional): Pen to draw border around the marker symbol. Defaults to None. Inherits charts default plot configuration.
        """
        ...

    @abstractmethod
    def set_data(self, id: str, *args, **kargs):
        """Updates the plot with provided data \n
        set_data(id, x, y):x, y: array_like coordinate values
        set_data(id, y): y values only – x will be automatically set to range(len(y))
        set_data(id, x=x, y=y): x and y given by keyword arguments
        set_data(id, ndarray(N,2)): single numpy array with shape (N, 2), where x=data[:,0] and y=data[:,1]

        Args:
            id (str): Unique id of the plot in the chart
        """
        ...

    @property
    @abstractmethod
    def title(self) -> str:
        """Title of the chart (get/set)"""
        ...

    @title.setter
    @abstractmethod
    def title(self, val: str):
        ...


def make_color(color: Color) -> Any:
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


def make_pen(config: PenConfig) -> Any:
    """Creates a QPen from a pen configuration"""
    if "style" in config:
        config["style"] = config["style"].value
    if "color" in config:
        config["color"] = make_color(config["color"])
    return pg.mkPen(**config)


def merge(base: dict, new: dict) -> dict:
    """Dictionary merge"""
    for key in new:
        if key in base:
            if isinstance(base[key], dict) and isinstance(new[key], dict):
                merge(base[key], new[key])
                continue
        base[key] = new[key]
    return base


def merge_config(base: ChartConfig, *args: Optional[ChartConfig]) -> ChartConfig:
    """Merges multiple config into one"""
    result = copy.deepcopy(base)
    for config in args:
        if config is not None:
            result = merge(result, config)
    return result


class ChartsWidget:
    """A Widget to display or record animation of multiple charts as rows."""

    class __ChartWidgetImpl(ChartWidget):
        def __init__(
            self, id: str, win: pg.GraphicsLayoutWidget, config: ChartConfig, title=None
        ) -> None:
            self.__chart: pg.PlotItem = win.addPlot(name=id, title=title)
            self.__chart.setPreferredHeight(config["chart_height"])
            self.__config = config
            legend = self.__chart.addLegend(offset=None)
            vb: pg.ViewBox = win.addViewBox()
            vb.setFixedWidth(config["legend_width"])
            legend.setParentItem(vb)
            self.__chart.setClipToView(True)

            if "x_axis" in config:
                xconfig = config["x_axis"]
                if "log" in xconfig:
                    self.__chart.setLogMode(x=xconfig["log"])
                if "label" in xconfig:
                    self.__chart.setLabel("bottom", xconfig["label"])
                if "range" in xconfig:
                    self.__chart.setXRange(*xconfig["range"])

            if "y_axis" in config:
                yconfig = config["y_axis"]
                if "log" in yconfig:
                    self.__chart.setLogMode(y=yconfig["log"])
                if "label" in yconfig:
                    self.__chart.setLabel("left", yconfig["label"])
                if "range" in yconfig:
                    min, max = yconfig["range"]
                    self.__chart.setYRange(min, max)

            self.__plots: Dict[str, pg.PlotDataItem] = {}

        def __add__plot(self, id: str, name: str, config: PlotConfig):

            pen = config.pop("pen")

            if pen is not None:
                pen = make_pen(pen)

            if "symbol" in config:
                config["symbol"] = config["symbol"].value

            if "symbolPen" in config:
                spen = config.pop("symbolPen")
                spen = make_pen(spen)
                config["symbolPen"] = spen

            if "symbolBrush" in config:
                sbrush = config.pop("symbolBrush")
                sbrush = make_color(sbrush)
                config["symbolBrush"] = sbrush

            self.__plots[id] = self.__chart.plot(pen=pen, name=name, **config)

        def add_line(
            self,
            id: str,
            name: str,
            color: Color = None,
            width: float = None,
            style: PenStyle = None,
        ):
            config = merge_config(self.__config)

            if color is not None:
                config["pen"]["color"] = color

            if width is not None:
                config["pen"]["width"] = width

            if style is not None:
                config["pen"]["style"] = style

            self.__add__plot(id, name, config=config)

        def add_scatter(
            self,
            id: str,
            name: str,
            size: float = None,
            color: Color = None,
            symbol: SymbolType = None,
            border: PenConfig = None,
        ):
            config = merge_config(DEFAULT_SYMBOL, self.__config) | {"pen": None}

            if size is not None:
                config["symbolSize"] = size

            if color is not None:
                config["symbolBrush"] = color

            if symbol is not None:
                config["symbol"] = symbol

            if border is not None:
                config["symbolPen"] = config["symbolPen"] | border

            self.__add__plot(id, name, config=config)

        def set_data(self, id: str, *args, **kargs):
            self.__plots[id].setData(*args, **kargs)

        @property
        def title(self) -> str:
            return self.__chart.title

        @title.setter
        def title(self, value: str):
            self.__chart.setTitle(value)

    def __init__(
        self,
        save_file: str = None,
        size: Tuple[int, int] = None,
        title: str = None,
        border=None,
        antialias: Optional[bool] = None,
        foreground: Color = None,
        background: Color = None,
        default_chart_config: ChartConfig = None,
    ):
        """Initializes a charts widget

        Args:
            save_file (str, optional): Name of the animation file. If not empty, hides UI. Defaults to None.
            size (Tuple[int,int], optional):(width,height) of the charts window. Defaults to (1000,1000).
            title (str, optional): Title of charts window. Defaults to None.
            border (Any, optional): Border between charts. Defaults to None.
            antialias (Optional[bool], optional): Use antialiasing. If true, smooth visuals and slower refresh rate. Defaults to None.
            foreground (Color, optional): General foreground color (text,ie). Defaults to None.
            background (Color, optional): General background color. Defaults to None.
            default_chart_config (ChartConfig, optional): Default chart configuration. Defaults to None.
        """
        self.__save = save_file is not None
        self.__init_size = DEFAULT_WINDOW_SIZE if size is None else size
        self.__app = pg.mkQApp(title)
        if antialias is not None:
            pg.setConfigOptions(antialias=antialias)
        if foreground is not None:
            pg.setConfigOption("foreground", make_color(foreground))
        if background is not None:
            pg.setConfigOption("background", make_color(background))
        self.__win = pg.GraphicsLayoutWidget(
            show=not self.__save, size=self.__init_size, title=title, border=border
        )
        self.__win.mouseEnabled = False
        self.__defult_chart_config = merge_config(DEFAULT_CHART, default_chart_config)
        self.__charts: Dict[str, ChartsWidget] = {}
        if self.__save:
            self.__ffmpeg_process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s=f"{self.__init_size[0]}x{self.__init_size[1]}",
                )
                .output(save_file, vcodec="libx264", pix_fmt="yuv420p")
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

    def add_chart(
        self,
        id: str,
        title: str = None,
        height: int = None,
        legend_width: int = None,
        x_label: str = None,
        x_range: Tuple[float, float] = None,
        x_log: bool = None,
        y_label: str = None,
        y_range: Tuple[float, float] = None,
        y_log: bool = None,
        default_plot_config: PlotConfig = None,
    ) -> ChartWidget:
        """Add a chart with legend for plots as a row

        Args:
            id (str): Unique id
            title (str, optional): Title to display on the top. Defaults to None.
            height (int, optional): Prefred height. Defaults to None.
            legend_width (int, optional):Legend width. Defaults to None.
            x_label (str, optional): Label text to display under x axis. Defaults to None.
            x_range (Tuple[float,float], optional): Constant range(min,max value) of x axis. Defaults to None.
            x_log (bool, optional): Use logarithmic scale to display x axis. Defaults to None.
            y_label (str, optional): Label text to display next to y axis. Defaults to None.
            y_range (Tuple[float,float], optional): Constant range(min,max value) of y axis. Defaults to None.
            y_log (bool, optional): Use logarithmic scale for y axis. Defaults to None.
            default_plot_config (PlotConfig, optional): Default plot configuration. Defaults to None. Inherits from chart default chart configuration.

        Returns:
            ChartWidget: Chart widget
        """
        config: ChartConfig = merge_config(
            self.__defult_chart_config, default_plot_config
        )
        if height is not None:
            config["chart_height"] = height
        if legend_width is not None:
            config["legend_width"] = legend_width
        if x_label is not None:
            config["x_axis"]["label"] = x_label
        if x_log is not None:
            config["x_axis"]["log"] = x_log
        if x_range is not None:
            config["x_axis"]["range"] = x_range
        if y_label is not None:
            config["y_axis"]["label"] = y_label
        if y_log is not None:
            config["y_axis"]["log"] = y_log
        if y_range is not None:
            config["y_axis"]["range"] = y_range
        self.__charts[id] = ChartsWidget.__ChartWidgetImpl(
            id, self.__win, config, title
        )
        self.__win.nextRow()
        return self.__charts[id]

    def render(self):
        """Renders chart changes to UI or animation file."""
        self.__app.processEvents()
        if self.__save:
            qimage = self.__win.grab().toImage()
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

            self.__ffmpeg_process.stdin.write(qimage.constBits().tobytes())

    @property
    def is_hidden(self) -> bool:
        """Checks if UI is hidden. Always true when save file is defined."""
        return self.__win.isHidden()

    def close(self):
        """Closes the UI or finishes animation file recording."""
        if self.__save:
            self.__ffmpeg_process.stdin.close()
            self.__ffmpeg_process.wait()

        if self.is_hidden:
            pg.exit()
        else:
            pg.exec()  # display till user closes the UI
