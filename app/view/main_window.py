# coding:utf-8
from PySide6.QtCore import Qt, QUrl, QRect, QSize, QTimer, QEventLoop ,QTranslator
from PySide6.QtGui import QIcon, QDesktopServices, QPainter, QBrush, QPen, QColor
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout
from qfluentwidgets import (NavigationItemPosition, MessageBox, setTheme, Theme, FluentWindow,
                            NavigationAvatarWidget, qrouter, SubtitleLabel, setFont, InfoBadge,FluentTranslator,
                            InfoBadgePosition, FluentBackgroundTheme, SplashScreen, PrimaryPushButton, SystemThemeListener, isDarkTheme)

from qfluentwidgets import FluentIcon as FIF
from app.view.NN_construct import NNConstruct
from app.view.NN_evaluate import NNEvaluate
from app.view.settings import Settings
from app.common.signal_bus import signalBus
import torch
from app.common.NN import NN
from app.common.config import cfg
from app.common.NN import Datasets
from app.resource import resource_rc
class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)

        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)

        # 必须给子界面设置全局唯一的对象名
        self.setObjectName(text.replace(' ', '-'))




class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.__initWindow()
        self.themeListener = SystemThemeListener(self)
        self.translator = QTranslator()
        self.translator.load(cfg.get(cfg.language).value, 'GUI', '.', ":/GUI/i18n")
        self.fluentTranslator = FluentTranslator(cfg.get(cfg.language).value)
        QApplication.instance().installTranslator(self.translator)
        QApplication.instance().installTranslator(self.fluentTranslator)
        # create sub interface
        self.device = torch.device("cuda" if cfg.useCUDA.value else "cpu")
        self.dataType = torch.cuda.FloatTensor if cfg.useCUDA.value else torch.FloatTensor
        self.dataset = Datasets(cfg.dataset.value)
        self.model = NN(self.dataset.inputShape, self.device)
        self.model.loadModel(cfg.modelFolder.value + '/' + cfg.model.value)
        self.NNConstructInterface = NNConstruct(self)
        self.NNEvaluateInterface = NNEvaluate(self)
        self.settingInterface = Settings(self)
        
        # enable acrylic effect
        self.navigationInterface.setAcrylicEnabled(True)

        self.__connectSignalToSlot()

        # add items to navigation interface
        self.__initNavigation()
        self.splashScreen.finish()

        # start theme listener
        self.themeListener.start()

    def __connectSignalToSlot(self):
        signalBus.micaEnableChanged.connect(self.setMicaEffectEnabled)
        signalBus.deviceChanged.connect(self.setDevice)
        signalBus.datasetChanged.connect(self.setDataset)
        signalBus.languageChanged.connect(self.setLanguage)


    def __initNavigation(self):
        # add navigation items
        self.addSubInterface(self.NNConstructInterface, FIF.ALBUM, self.tr('NN Construst'))
        self.addSubInterface(self.NNEvaluateInterface, FIF.ALBUM, self.tr('NN Evaluate'))
        self.addSubInterface(self.settingInterface, FIF.SETTING, self.tr('Settings'))

        
    def __initWindow(self):
        self.resize(960, 780)
        self.setMinimumWidth(760)
        self.setWindowIcon(QIcon(':/GUI/images/logo.png'))
        self.setWindowTitle(self.tr('NN Tools'))

        self.setMicaEffectEnabled(cfg.get(cfg.micaEnabled))

        # create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(106, 106))
        self.splashScreen.raise_()

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        self.show()
        QApplication.processEvents()


    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'splashScreen'):
            self.splashScreen.resize(self.size())

    def closeEvent(self, e):
        self.themeListener.terminate()
        self.themeListener.deleteLater()
        super().closeEvent(e)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

        # retry
        if self.isMicaEffectEnabled():
            QTimer.singleShot(100, lambda: self.windowEffect.setMicaEffect(self.winId(), isDarkTheme()))

    def setDevice(self):
        # print(cfg.useCUDA.value)
        self.device = torch.device("cuda" if cfg.useCUDA.value else "cpu")
        self.dataType = torch.cuda.FloatTensor if cfg.useCUDA.value else torch.FloatTensor
        self.NNEvaluateInterface.updateDevice(self.device, self.dataType)
    
    def setDataset(self):
        self.dataset = Datasets(cfg.dataset.value)
        print(self.dataset.classes)
        print(len(self.dataset.classes))
        self.NNEvaluateInterface.updateDataset(self.dataset)
        self.NNConstructInterface.updateDataset(self.dataset)

    def setModel(self, modelPth):
        datasetName = modelPth.split('.')[-2]
        if datasetName != cfg.dataset.value:
            cfg.set(cfg.dataset, datasetName)
            self.setDataset()
        cfg.set(cfg.model, modelPth)
        self.model = NN(self.dataset.inputShape, self.device)
        self.model.loadModel(cfg.modelFolder.value + '/' + modelPth)
        
        self.NNEvaluateInterface.updateModel(self.model)
        self.NNConstructInterface.updateModel(self.model)

        

    def setLanguage(self):
        # self.translator.load(f':/GUI/i18n/GUI.{cfg.get(cfg.language).value.name()}.qm')
        QApplication.instance().removeTranslator(self.translator)
        QApplication.instance().removeTranslator(self.fluentTranslator)
        self.fluentTranslator = FluentTranslator(cfg.get(cfg.language).value)
        self.translator = QTranslator()
        self.translator.load(cfg.get(cfg.language).value, 'GUI', '.', ":/GUI/i18n")
        QApplication.instance().installTranslator(self.translator)
        QApplication.instance().installTranslator(self.fluentTranslator)
        self.update()
        

