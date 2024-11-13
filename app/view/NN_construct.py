from PySide6.QtCore import Qt, QUrl, QEvent, QThread, QRect
from PySide6.QtGui import QDesktopServices, QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QFormLayout, QGridLayout, QSizePolicy, QStackedWidget
from matplotlib import texmanager
from qfluentwidgets import (PushButton, ScrollArea, LineEdit, ToolButton, FluentIcon, Action, MessageBox,Flyout,FlyoutAnimationType,
                            isDarkTheme, ComboBox, Theme, ToolTipFilter, TitleLabel, CaptionLabel,InfoBarIcon,Pivot,
                            StrongBodyLabel, MessageBoxBase, toggleTheme, SubtitleLabel, IndeterminateProgressBar, CommandBar, qconfig)
from qfluentwidgets import FluentIcon as FIF
from app.common.NN import NN
from app.view.content_widgets import Content
from app.common.config import cfg
from collections import OrderedDict
import torch.nn as nn
class LayerSetting(QWidget):
    def __init__(self, layerType: str, parent = None):
        super().__init__(parent)
        self.layerType = layerType
        self.formLayout = QFormLayout(self)
        self.formData = None
        self.kargsMap = {
            'kernel_size' : self.tr('Kernel Size') ,
            'out_channels' : self.tr('Out Channels'),
            'stride' : self.tr('Stride'),
            'padding' : self.tr('Padding'),
            'out_features' : self.tr('Out Features')
        }
        if self.layerType == 'Conv2d':
            parameters = ['kernel_size', 'out_channels', 'stride', 'padding']
            self.formData = dict(zip(parameters, [LineEdit(self) for _ in range(len(parameters))]))
            for parameter in parameters:
                self.formLayout.addRow(self.kargsMap[parameter], self.formData[parameter])    
        elif self.layerType == 'MaxPool2d':
            parameters = ['kernel_size', 'stride', 'padding']
            self.formData = dict(zip(parameters, [LineEdit(self) for _ in range(len(parameters))]))
            for parameter in parameters:
                self.formLayout.addRow(self.kargsMap[parameter], self.formData[parameter])
        elif self.layerType == 'Flatten':
            pass
        elif self.layerType == 'Linear':
            parameters = ['out_features']
            self.formData = dict(zip(parameters, [LineEdit(self) for _ in range(len(parameters))]))
            for parameter in parameters:
                self.formLayout.addRow(self.kargsMap[parameter], self.formData[parameter])

    def getSetting(self):
        try:
            return dict(zip(self.formData.keys(), map(lambda x: self.mapValues(x.text()), self.formData.values())))
        except:
            return None

    def mapValues(self, value : str):
        try:
            return int(value)
        except:
            return None
        

class LayerChoose(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.pivot = Pivot(self)
        self.stackedWidget = QStackedWidget(self)
        self.vBoxLayout = QVBoxLayout(self)

        self.Conv2dInterface = LayerSetting('Conv2d', self)
        self.MaxPool2dInterface = LayerSetting('MaxPool2d', self)
        self.FlattenInterface = LayerSetting('Flatten', self)
        self.LinearInterface = LayerSetting('Linear', self)

        # 添加标签页
        self.addSubInterface(self.Conv2dInterface, 'Conv2dInterface', 'Conv2d')
        self.addSubInterface(self.MaxPool2dInterface, 'MaxPool2dInterface', 'MaxPool2d')
        self.addSubInterface(self.FlattenInterface, 'FlattenInterface', 'Flatten')
        self.addSubInterface(self.LinearInterface, 'LinearInterface', 'Linear')

        # 连接信号并初始化当前标签页
        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.Conv2dInterface)
        self.pivot.setCurrentItem(self.Conv2dInterface.objectName())

        self.vBoxLayout.setContentsMargins(10, 0, 10, 10)
        self.vBoxLayout.addWidget(self.pivot, 0, Qt.AlignLeft)
        self.vBoxLayout.addWidget(self.stackedWidget)
        # self.resize(400, 400)
    def addSubInterface(self, widget: QWidget, objectName: str, text: str):
        widget.setObjectName(objectName)
        # widget.setAlignment(Qt.AlignLeft)
        self.stackedWidget.addWidget(widget)

        # 使用全局唯一的 objectName 作为路由键
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())
class AddLayer(MessageBoxBase):
    """ Custom message box """
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.lineEdit = LineEdit()
        # self.formLayout = QFormLayout()
        # #self.lineEdit.setValidator(QRegExpValidator(QRegExp('[0-9]*')))

        # self.comboBox = ComboBox()
        # self.comboBox.addItems(["Conv2d", "MaxPool2d", "Linear", "Flatten"])
        
        # self.formLayout.addRow(SubtitleLabel('Layer Type'), self.comboBox)
        self.layerChoose = LayerChoose(self)
        # 将组件添加到布局中
        self.viewLayout.addWidget(self.layerChoose)

        # 设置对话框的最小宽
        self.widget.setMinimumWidth(350)
class SaveModel(MessageBoxBase):
    """ Custom message box """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lineEdit = LineEdit()
        self.formLayout = QFormLayout()
        #self.lineEdit.setValidator(QRegExpValidator(QRegExp('[0-9]*')))
        self.formLayout.addRow(self.tr('Model Name'), self.lineEdit)
        self.viewLayout.addLayout(self.formLayout)
class ShowNNStructre(QFrame):
    def __init__(self, name, nnStructure, parent=None):
        super(ShowNNStructre, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName(name)
        self.textColor = Qt.white if isDarkTheme() else Qt.black

        self.nnStructure = nnStructure
        qconfig.themeChanged.connect(self.changeTextColor)
    def drawSquare(self, painter : QPainter, x, y, width, color : QColor, borderColor : QColor = Qt.black, borderWidth = 1):
        qconfig.themeChanged.connect(self.changeTextColor)
    def paintEvent(self, event):
        painter = QPainter(self)
        
        self.drawNN(painter, self.nnStructure)
    def drawSquare(self, painter : QPainter, x, y, width, color : QColor, borderColor : QColor = Qt.black, borderWidth = 1):
        painter.fillRect(x, y, width, width, color)
        if borderColor != None:
            painter.setPen(QPen(borderColor, borderWidth))
            painter.drawRect(x, y, width, width)
    
    def drawConvLayer(self, painter : QPainter, x, y, layerWidth, layerNum = 3, gap = 10, omit = False):
        if not omit:
            for i in range(layerNum):
                self.drawSquare(painter, x, y, layerWidth, [Qt.gray, Qt.darkGray][i%2], borderColor=None)
                x += gap
                y += gap
            return x-gap, y-gap, layerWidth
        else:
            for i in range(4):
                self.drawSquare(painter, x, y, layerWidth, [Qt.gray, Qt.darkGray][i%2], borderColor=None)
                if i == 1:
                    x += layerWidth/2
                    y += layerWidth/2
                    for j in range(3):
                        x += gap
                        y += gap
                        painter.setPen(QPen(Qt.black, 2))
                        painter.drawEllipse(x, y, 1, 1)
                    x -= layerWidth/2
                    y -= layerWidth/2

                x += gap
                y += gap

    def changeTextColor(self):
        self.textColor = Qt.white if isDarkTheme() else Qt.black

    def drawConvLayerConnection(self, painter : QPainter, x1, y1, x2, y2, width = 20):
        self.drawSquare(painter, x1, y1, width, QColor(50, 50, 50))
        painter.drawLine(x1, y1, x2, y2)
        painter.drawLine(x1 + width, y1, x2, y2)
        painter.drawLine(x1, y1 + width, x2, y2)
        painter.drawLine(x1 + width, y1 + width, x2, y2)


    def drawNN(self, painter : QPainter, model: OrderedDict):
        layers = [i for i in model if 'ReLU' not in i]
        sectionWidth = self.parent().width() / (len(layers) + 2.5)
        x = 0
        y = 50
        widthRatio = 1
        convLayerConnections = []
        fcLayerConnections = []
        translate = {'Conv2d': 'Feature Map', 'MaxPool2d': 'Feature Map', 'Linear': 'Hidden Units', 'Flatten': 'Hidden Units'}
        for section, layer in enumerate(layers):
            if section == 0:
                painter.setPen(QPen(self.textColor, 2))
                painter.drawText(QRect(x, 0, sectionWidth, 50), Qt.AlignCenter, 'Input' + '\n' + str(model[layer]['input_shape'][1:]))
                inputLayerNum = model[layer]['input_shape'][1]
                inputWidth = sectionWidth/(1 + .1 * (inputLayerNum - 1))
                if inputLayerNum == 1:
                    inputWidth -= 20
                widthRatio = inputWidth / model[layer]['input_shape'][2]
                inputGap = .1 * inputWidth
                endx, endy, width = self.drawConvLayer(painter, x, y, inputWidth, layerNum=inputLayerNum, gap = inputGap)
                if 'Conv2d' in layer or 'MaxPool2d' in layer:
                    convLayerConnections.append([endx, endy, width, 'Input'])
                x += sectionWidth + 5
            painter.setPen(QPen(self.textColor, 2))
            if section == len(layers) -1 :
                painter.drawText(QRect(x, 0, sectionWidth, 50), Qt.AlignCenter, 'Output' + '\n' + str(model[layer]['output_shape'][1:]))
            else:
                painter.drawText(QRect(x, 0, sectionWidth, 50), Qt.AlignCenter, translate[layer.split('-')[0]] + '\n' + str(model[layer]['output_shape'][1:]))
            # w + (L-1) * g = W
            # g = (W-w)/(L-1)
            layerNum = model[layer]['output_shape'][1]
            omit = False
            if  len(model[layer]['output_shape']) > 2:
                layerWidth = widthRatio * model[layer]['output_shape'][2]  
            else: 
                layerWidth = 10
            gap = (sectionWidth - layerWidth) / (layerNum - 1)
            endx, endy, width = self.drawConvLayer(painter, x, y, layerWidth, layerNum=layerNum, gap = gap, omit=omit)
            if 'Conv2d' in layer or 'MaxPool2d' in layer:
                convLayerConnections.append([endx, endy, width, layer])
            else:
                fcLayerConnections.append([endx, endy, width, layer])
            x += sectionWidth + 5
        convLayerNum = len(convLayerConnections)
        fcLayerNum = len(fcLayerConnections)
        connections = convLayerConnections + fcLayerConnections
        for i in range(len(connections) -1 ):
            width1 = connections[i][2]
            width2 = connections[i+1][2]
            centerX1 = connections[i][0] + width1/2
            centerY1 = connections[i][1] + width1/2

            centerX2 = connections[i+1][0] + width2/2
            centerY2 = connections[i+1][1] + width2/2
            width = .2*width1
            if i < convLayerNum - 1:
                if i % 2 == 0:
                    self.drawConvLayerConnection(painter, centerX1 - width/2, centerY1- .25 * width1 - width/2, centerX2, centerY2 -.25 * width2, width = width)
                else:
                    self.drawConvLayerConnection(painter, centerX1 - width/2, centerY1 + .25 * width1 - width/2, centerX2, centerY2 + .25 * width2, width = width)
            painter.setPen(QPen(self.textColor, 2))
            painter.drawText(QRect(centerX1, sectionWidth + 100, centerX2 - centerX1, 50), Qt.AlignCenter, connections[i+1][3])

    def updateStructure(self, structure):
        self.nnStructure = structure
         
        

        
    


class NNConstruct(Content):
    """ Text interface """
    def __init__(self, parent=None):
        super(NNConstruct, self).__init__(
            title=self.tr('NNConstruct'),
            subtitle='test',
            parent=parent
        )
        self.layerMap = {
            'Conv2d': nn.Conv2d, 
            'MaxPool2d': nn.MaxPool2d, 
            'ReLU': nn.ReLU, 
            'Linear': nn.Linear, 
            'Dropout': nn.Dropout, 
            'Flatten': nn.Flatten
        }
        self.device = parent.device
        self.dataType = parent.dataType
        self.dataset = parent.dataset
        self.inputShape = parent.dataset.inputShape
        self.model : NN = parent.model
        self.currentShape = self.model.outputShape
        if self.currentShape is None:
            self.currentShape = self.inputShape
        #self.layers = [i for i in self.nnStructure if 'ReLU' not in i]
        self.__initWidgets()
        self.__initLayout()
        self.__connectSignalsToSlot()


    def __initWidgets(self):
        self.canvas = ShowNNStructre('1', self.model.NNStructure, self)
        self.addLayerBtn = PushButton(FIF.ADD, self.tr('Add Layer'), self)
        self.newModelBtn = PushButton(FIF.LAYOUT, self.tr('New Model'), self)
        self.trainBtn = PushButton(FIF.ACCEPT, self.tr('Train Model'), self)
        self.saveBtn = PushButton(FIF.SAVE, self.tr('Save Model'), self)
        self.toolBar.addTools([self.addLayerBtn, self.newModelBtn, self.trainBtn, self.saveBtn])

    def __initLayout(self):
        
        self.inputLayout = QGridLayout()

        self.inputLayout.addWidget(self.canvas, 0, 0, 1 ,-1)
        self.addContentLayout(self.inputLayout)
        
    
    def __connectSignalsToSlot(self):
        self.addLayerBtn.clicked.connect(self.addLayer)
        self.newModelBtn.clicked.connect(self.newModel)
        self.trainBtn.clicked.connect(self.trainModel)
        self.saveBtn.clicked.connect(self.saveModel)
        

    
    def addLayer(self):
        self.AddLayer = AddLayer(parent=self)
        if self.AddLayer.exec():
            print('Yes button is pressed')
            # self.nnStructure[self.layers[0]] = testStructure[self.layers[0]]
            # self.layers.pop(0)
            temp = self.AddLayer.layerChoose.stackedWidget.currentWidget()
            layerType = temp.layerType
            print(temp.getSetting())
            kargs = temp.getSetting()
            if layerType == 'Conv2d':
                kargs['in_channels'] = self.currentShape[0]
            elif layerType == 'Linear':
                kargs['in_features'] = self.currentShape[0]
            try:
                newLayer = self.layerMap[layerType](**kargs)
                
            except:
                newLayer = self.layerMap[layerType]()
            
            self.model.append(newLayer)
            self.model.to(self.device)
            tempStructure = self.model.NNStructure
            self.currentShape = tempStructure[list(tempStructure.keys())[-1]]['output_shape'][1:]
            print(self.currentShape)
            #self.layers = [i for i in self.nnStructure if 'ReLU' not in i]
            self.canvas.updateStructure(tempStructure)
        else:
            print('Cancel button is pressed')
        
    def newModel(self):
        self.model = NN(self.inputShape, self.device)
        self.currentShape = self.inputShape
        self.canvas.updateStructure(OrderedDict())

    def trainModel(self):
        self.model.train(
            learning_rate=0.001,
            batch_size=64,
            num_epochs=5,
            train_dataset=self.dataset.trainData,
            test_dataset=self.dataset.testData
        )

    def saveModel(self):
        self.SaveModel = SaveModel(parent=self)
        if self.SaveModel.exec():
            print('Yes button is pressed')
            modelName = self.SaveModel.lineEdit.text()
            self.model.saveModel(cfg.modelFolder.value + '/' + modelName + '.' + self.dataset.name + '.pt')
        else:
            print('Cancel button is pressed')
        
        #self.model.saveModel('./app/data/model3.pt')
    
    def updateModel(self, model):
        self.model = model
        self.model.to(self.device)
        self.canvas.updateStructure(self.model.NNStructure)
        self.update()

    def updateDataset(self, dataset):
        self.dataset = dataset
        self.inputShape = self.dataset.inputShape
        self.update()

