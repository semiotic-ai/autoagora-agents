# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
import ffmpeg
from typing import Any, Dict, Optional, Tuple,TypedDict
from pyqtgraph.Qt import QtCore, QtWidgets,QtGui
import pyqtgraph as pg

class DownSampleMethod(Enum):
    subsample='subsample'
    mean='mean'
    peak='peak'

class PenConfig(TypedDict,total=False):
    width:float
    style:Any
    color:Any
    
class PlotConfig(TypedDict,total=False):
    symbolBrush:Any
    symbolSize:float
    symbolPen:PenConfig
    autoDownsample:bool
    downsample:int
    downsampleMethod:DownSampleMethod
    pen:Optional[PenConfig]

class AxisConfig(TypedDict,total=False):
    label:str
    range:Tuple[float,float]
    log:bool

class ChartConfig(PlotConfig,total=False):
    chart_height:int
    legend_width:int
    x_axis:AxisConfig
    y_axis:AxisConfig
    

DEFAULT_PEN:PenConfig = {
    "width":1,
    "style":QtCore.Qt.SolidLine,
    "color":"gray",
}

DEFAULT_CHART:ChartConfig = {
    "chart_height":300,
    "legend_width":300,
    "pen":DEFAULT_PEN
}    

DEFAULT_AXIS:AxisConfig = {
    "log":False
}

DEFAULT_WINDOW_SIZE = (1000, 1000)

class ChartWidget(ABC):

    @abstractmethod
    def addPlot(self,id:str,name:str,config:PlotConfig):...

    @abstractmethod
    def setPlotData(self,id:str,*args):...        
    
    @property
    @abstractmethod
    def title(self) -> str:...
    
    @title.setter
    @abstractmethod
    def title(self, val:str):...

class ChartsWidget:
    
    class __ChartWidgetImpl(ChartWidget):
        def __init__(self,id:str,win:pg.GraphicsLayoutWidget,config:ChartConfig,title=None) -> None:
            self.__chart:pg.PlotItem = win.addPlot(name=id,title=title)
            self.__chart.setPreferredHeight(config["chart_height"]) 
            self.__config = config
            legend = self.__chart.addLegend(offset=None)
            vb:pg.ViewBox = win.addViewBox()
            vb.setFixedWidth(config["legend_width"])
            legend.setParentItem(vb)
            self.__chart.setClipToView(True)
            
            xlog = False
            ylog = False 
            
            if "x_axis" in config:
                xconfig = DEFAULT_AXIS | config["x_axis"]
                xlog = xconfig["log"]
                if "label" in xconfig:
                    self.__chart.setLabel("bottom",xconfig["label"])
                if "range" in xconfig:
                    self.__chart.setXRange(*xconfig["range"])
            
            if "y_axis" in config:
                yconfig = DEFAULT_AXIS | config["y_axis"]
                ylog = yconfig["log"]
                if "label" in yconfig:
                    self.__chart.setLabel("left",yconfig["label"])
                if "range" in yconfig:
                    self.__chart.setYRange(*yconfig["range"])

            self.__chart.setLogMode(xlog,ylog)
            
            self.__plots:Dict[str,pg.PlotDataItem] = {}            
        
        def addPlot(self,id:str,name:str,config:PlotConfig):
            config = self.__config | config
            pen = config.pop("pen")
                
            if pen is not None:
                pen = pg.mkPen(**pen)
        
            self.__plots[id] = self.__chart.plot(
                pen = pen,
                name = name,                
                **config
            ) 
                
        def setPlotData(self,id:str,*args,**kargs):
            self.__plots[id].setData(*args,**kargs)
        
        @property
        def title(self)->str:
            return self.__chart.title

        @title.setter
        def title(self,value:str):
            self.__chart.setTitle(value)

    def __init__(self,save_file:str=None, size=None, title=None, border=None,antialias:Optional[bool]=None,foreground=None,background=None,default_chart_config=DEFAULT_CHART):
        self.__save = save_file is not None
        self.__init_size = DEFAULT_WINDOW_SIZE if size is None else size
        self.__app =  pg.mkQApp(title)
        if antialias is not None:
            pg.setConfigOptions(antialias=antialias)
        if foreground is not None:
            pg.setConfigOptions("foreground",foreground)
        if background is not None:
            pg.setConfigOption("background",background)        
        self.__win = pg.GraphicsLayoutWidget(show=not self.__save, size=self.__init_size, title=title, border=border)
        self.__win.mouseEnabled = False
        self.__defult_chart_config = DEFAULT_CHART | default_chart_config
        self.__charts:Dict[str,ChartsWidget] = {}        
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

    def addChart(self,id:str,title=None,config=DEFAULT_CHART) -> ChartWidget:
        config:ChartConfig = self.__defult_chart_config | config
        self.__charts[id] = ChartsWidget.__ChartWidgetImpl(id,self.__win,config,title)
        self.__win.nextRow()
        return self.__charts[id]
    
    def render(self):
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