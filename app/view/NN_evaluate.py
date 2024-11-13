from PySide6.QtCore import Qt, QUrl, QEvent, QThread, QRect, QSize, QRectF, QPoint, QPointF
from PySide6.QtGui import QDesktopServices, QPainter, QPen, QColor, QBrush, QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QFormLayout, QGridLayout, QSizePolicy, QGraphicsView
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QBarSeries, QBarSet, QPieSeries, QPieSlice, QAbstractBarSeries, QBarCategoryAxis, QValueAxis, QChartView
from qfluentwidgets import (PushButton, ImageLabel, isDarkTheme,  qconfig, TitleLabel, ComboBox)
from app.view.content_widgets import Content, WidgetCard
from app.common.signal_bus import signalBus
from collections import OrderedDict
from PIL import Image, ImageQt
from app.common.NN import get_cam_mask, Datasets, NN
from app.common.config import cfg
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


class PaintBoard(QFrame):
    def __init__(self, name, parent=None):
        super(PaintBoard, self).__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName(name)
        self.lineColor =  Qt.white if isDarkTheme() else Qt.black
        qconfig.themeChanged.connect(self.changeLineColor)
        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)
        self.image = None

        '''
            要想将按住鼠标后移动的轨迹保留在窗体上
            需要一个列表来保存所有移动过的点
        '''
        self.pos_xy = []

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        
        pen = QPen(self.lineColor, 10, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        '''
            首先判断pos_xy列表中是不是至少有两个点了
            然后将pos_xy中第一个点赋值给point_start
            利用中间变量pos_tmp遍历整个pos_xy列表
                point_end = pos_tmp

                判断point_end是否是断点，如果是
                    point_start赋值为断点
                    continue
                判断point_start是否是断点，如果是
                    point_start赋值为point_end
                    continue

                画point_start到point_end之间的线
                point_start = point_end
            这样，不断地将相邻两个点之间画线，就能留下鼠标移动轨迹了
        '''
        if self.image != None:
            painter.drawImage(0, 0, ImageQt.toqimage(self.image))
        else:
            painter.fillRect(0, 0, self.width(), self.height(), Qt.white if self.lineColor == Qt.black else Qt.black)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()
        

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
            调用update()函数在这里相当于调用paintEvent()函数
            每次update()时，之前调用的paintEvent()留下的痕迹都会清空
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)
        signalBus.canvasUpdate.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
            然后在绘画时判断一下是不是断点就行了
            是断点的话就跳过去，不与之前的连续
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def changeLineColor(self):
        self.lineColor = Qt.white if isDarkTheme() else Qt.black
        signalBus.canvasUpdate.emit()
    
    def getPixmap(self) -> Image.Image:
        pm = QPixmap(self.width(), self.height())

        self.render(pm)
        image = Image.fromqpixmap(pm).convert('L')

        return image
        # pm.save('test.png', 'png')  # 保存图片

    def importImage(self, img:Image.Image):
        self.pos_xy = []
        self.image = img.copy()
        signalBus.canvasUpdate.emit()
        self.update()

    def clear(self):
        self.pos_xy = []
        self.image = None
        signalBus.canvasUpdate.emit()
        self.update()



class NNEvaluate(Content):
    def __init__(self, parent = None):
        super(NNEvaluate, self).__init__(
            title= self.tr('NNEvaluate'),
            subtitle=cfg.dataset.value,
            parent=parent
        )
        self._parent = parent
        self.modelsPth = filter(
            lambda x : x.endswith('.pt'),
            next(os.walk(cfg.modelFolder.value))[-1]
        )
        self.device = parent.device
        self.dataType = parent.dataType
        self.dataset : Datasets = parent.dataset
        self.model : NN = parent.model
        self.model.eval()
        featureMapLayers = self.model.featureMapLayer()
        self.shapeSeq = [
            tuple(featureMapLayers[layer]['output_shape'][2:]) for layer in featureMapLayers
        ]
        
        self.outputList = None
        self.camLayer = self.model.lastConvLayer()
       
        self.__initWidget()
        self.__connectSignalToSlot()
        self.__initLayout()
        self.__initBarChart()
    
    def __initWidget(self):
        self.canvas = PaintBoard('canvas', self)
        self.canvas.setFixedSize(256, 256)
        self.clearBtn = PushButton(self.tr('clear'), self)
        self.layerChoice = ComboBox(self)
        self.layerChoice.addItems([layer for layer in self.model.featureMapLayer()])
        self.layerChoice.setCurrentIndex(self.camLayer)
        
        self.modelChoose = ComboBox(self)
        self.modelChoose.addItems(self.modelsPth)
        self.modelChoose.setCurrentIndex(0)
        self.importBtn = PushButton(self.tr('Import'), self)
        self.convResult = ImageLabel(self)
        self.maxPoolResult = ImageLabel(self)
        self.classActivateMap = ImageLabel(self)
        self.predicVal = TitleLabel()
        self.pout = QChartView()
        self.pout.setRenderHint(QPainter.Antialiasing)
        self.hiddenRes = [ImageLabel(self) for _ in range(6)]
        self.hiddenLayerCard = WidgetCard(self.tr("Hidden Layer"), self.hiddenRes, stretch=1, gap = 0)
        self.classActivateMapCard = WidgetCard(self.tr("Class Activate Map"), [self.classActivateMap, self.layerChoice], stretch=1, topLayout='V')
        self.outputCard = WidgetCard(self.tr("Output"),[self.predicVal, self.pout], topLayout='V')
        
        self.inputCard = WidgetCard(self.tr("Input"), [self.canvas, [self.clearBtn, self.importBtn]], topLayout='V')

        self.inputCard.setFixedWidth(276)
        self.outputCard.setFixedHeight(349)
        self.outputCard.setStyleSheet("width: 100%; height: 349px;")

        self.toolBar.addTools([self.modelChoose])
        
        

    def __connectSignalToSlot(self):
        self.layerChoice.currentIndexChanged.connect(self.setCamLayer)
        signalBus.canvasUpdate.connect(self.setOutput)
        self.clearBtn.clicked.connect(self.canvas.clear)
        self.importBtn.clicked.connect(self.handleImportBtnClick)
        qconfig.themeChanged.connect(self.changeChartTheme)

        self.modelChoose.currentIndexChanged.connect(
            lambda : self._parent.setModel(self.modelChoose.currentText()))
        
    def __initBarChart(self):
        #（1）创建图表和视图
        chart = QChart()
        chart.legend().setVisible(False)
        
        self.pout.setChart(chart)
        self.pout.setStyleSheet("background-color: transparent")
        self.pout.setContentsMargins(0, 0, 0, 0)
        chart.layout().setContentsMargins(0, 0, 0, 0)
        chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        chart.setBackgroundBrush(Qt.transparent)
   
        #（2）创建序列并添加数据
        self.outputSet = QBarSet('p')

        self.outputSet.append(np.round(self.outputList, 2).tolist())

        bSeries = QBarSeries()
        bSeries.append(self.outputSet)
        bSeries.setLabelsVisible(True)
        bSeries.setLabelsPosition(QAbstractBarSeries.LabelsPosition.LabelsOutsideEnd)
    
        chart.addSeries(bSeries)

    

        #（3）建立和设置坐标轴
        classes = self.dataset.classes
        axisX = QBarCategoryAxis()
        axisX.setTitleText(self.tr('Classes'))
        axisX.append(classes)
        chart.addAxis(axisX, Qt.AlignmentFlag.AlignBottom)
        bSeries.attachAxis(axisX)
        

        axisY = QValueAxis()
        axisY.setTitleText(self.tr('Possibility'))
        #axisY.setRange(0, 1)
        #axisY.setLabelsVisible(False)
        chart.addAxis(axisY, Qt.AlignmentFlag.AlignLeft)
        bSeries.attachAxis(axisY)
        
    def __initLayout(self):
        self.imageLayout = QGridLayout()
        self.imageLayout.addWidget(self.inputCard, 0, 0)
        self.imageLayout.addWidget(self.classActivateMapCard, 0, 1)
        self.imageLayout.addWidget(self.outputCard, 0, 2)
        
        self.imageLayout.addWidget(self.hiddenLayerCard, 1, 0, 1, 3)

        self.addContentLayout(self.imageLayout)
        
        self.setOutput()

    def changeChartTheme(self):
        self.pout.chart().setTheme(QChart.ChartTheme.ChartThemeDark if isDarkTheme() else QChart.ChartTheme.ChartThemeLight)
        self.pout.chart().setBackgroundBrush(Qt.transparent)

    def setCamLayer(self):
        self.camLayer = self.layerChoice.currentIndex()
        self.setOutput()

    def setOutput(self):
        #print(self.outputCard.height())
        cmap = plt.get_cmap('rainbow')
        self.model.eval()
        inputImg_ori = self.canvas.getPixmap()
        # if "EMNIST" in self.dataset.name:
        #     inputImg_ori.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)
        inputImg = inputImg_ori.resize(self.dataset.inputShape[1:])
        inputImg = np.array(inputImg).reshape(self.dataset.inputShape) / 255.
        camMask = get_cam_mask(self.model.model, self.model.model[self.camLayer], 
                               torch.as_tensor(inputImg.reshape((1, *self.dataset.inputShape))).type(self.dataType))
        camMask = cmap(camMask.reshape(self.dataset.inputShape[1:]))
        camMask = Image.fromarray((camMask*255).astype(np.uint8))
        camMask = camMask.resize((256, 256))
        camImg = Image.blend(inputImg_ori.convert("RGBA"), camMask.convert("RGBA"), 0.5)
        self.classActivateMap.setPixmap(ImageQt.toqpixmap(camImg))
        temp = inputImg
        # shapeSeq = [(28, 28), (14, 14), (14, 14), (7, 7), (7, 7), (3, 3)]

        maxHeight = self.dataset.inputShape[1] * 6
        for i in range(len(self.shapeSeq)):
            showNum = int(maxHeight/self.shapeSeq[i][0])
            temp = list(self.model.model)[i](torch.as_tensor(temp).type(self.dataType))
            outputImg = temp.cpu().detach().numpy()
            outputImg = cmap(outputImg[:showNum].reshape((self.shapeSeq[i][0]*showNum, self.shapeSeq[i][1])))
            outputImg = Image.fromarray((outputImg * 255).astype(np.uint8))
            w, h = outputImg.size
            outputImg = outputImg.resize((int(w*2), int(h*2)))
            self.hiddenRes[i].setPixmap(ImageQt.toqpixmap(outputImg))

        inpt = inputImg_ori.resize(self.dataset.inputShape[1:])
        inpt = np.array(inpt).reshape((1, *self.dataset.inputShape)) / 255.
        self.outputList = nn.functional.softmax(self.model.model(torch.as_tensor(inpt).type(self.dataType)), dim=-1).cpu().detach().numpy()[0]
        #print(self.outputList)
        self.predicVal.setText(self.tr('Predict:  ') + f'{self.dataset.classes[np.argmax(self.outputList)]}')
        try:
            for i, val in enumerate(np.round(self.outputList, 2).tolist()):
                self.outputSet.replace(i, val)
        except:
            pass

        #self._initBarChart()
        #self.convResult.setPixmap(result)
    
    def handleImportBtnClick(self):
        img = self.dataset.trainData[np.random.randint(0, len(self.dataset.trainData)-1)][0].numpy()
        img = Image.fromarray((img.reshape(self.dataset.inputShape[1:])*255).astype(np.uint8))
        img = img.resize(self.canvas.size().toTuple())
        # if 'EMNIST' in self.dataset.name:
        #     img = img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
        #print(self.dataset.data[0][0].numpy().shape)
        self.canvas.importImage(img)
        #print(img)
    

    def updateDevice(self, device, dataType):
        self.device = device
        self.dataType = dataType
        self.model.to(device)
        self.update()

    def updateDataset(self, dataset):
        self.dataset = dataset
        self.setSubTitle(dataset.name)
        self.__initBarChart()
        self.setOutput()
        self.update()

    def updateModel(self, model):
        self.model = model
        self.model.to(self.device)
        featureMapLayers = self.model.featureMapLayer()
        self.shapeSeq = [
            tuple(featureMapLayers[layer]['output_shape'][2:]) for layer in featureMapLayers
        ]
        self.outputList = None
        self.camLayer = self.model.lastConvLayer()
        self.layerChoice.clear()
        self.layerChoice.addItems([layer for layer in featureMapLayers])
        self.layerChoice.setCurrentIndex(self.camLayer)

        self.setOutput()
        self.update()


        

