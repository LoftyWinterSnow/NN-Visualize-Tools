from app.common.config import *
from PySide6.QtCore import Qt,QStandardPaths
from PySide6.QtGui import QDesktopServices, QPainter, QPen, QColor, QBrush, QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QFormLayout, QGridLayout, QSizePolicy, QGraphicsView
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QBarSeries, QBarSet, QPieSeries, QPieSlice, QAbstractBarSeries, QBarCategoryAxis, QValueAxis, QChartView
from qfluentwidgets import (ComboBoxSettingCard, SettingCardGroup, SwitchSettingCard, PushSettingCard, setThemeColor, 
                            toggleTheme, Theme, isDarkTheme, OptionsSettingCard, CustomColorSettingCard, FolderListSettingCard, setTheme, InfoBar)
from qfluentwidgets import FluentIcon as FIF

from app.common.signal_bus import signalBus
from app.view.content_widgets import Content
from app.common.config import cfg, CUDA_IS_AVAILABLE
from app.common.style_sheet import StyleSheet
class Settings(Content):
    def __init__(self, parent=None):
        super(Settings, self).__init__(
            title = "Settings",
            subtitle = '',
            showToolBar=False,
            parent = parent
        )
        self.useCUDA = SwitchSettingCard(
                configItem = cfg.useCUDA,
                icon = FIF.FLAG,
                title = "CUDA",
                content= self.tr("Choose whether to use CUDA"),

        )
        self.useCUDA.setEnabled(CUDA_IS_AVAILABLE)
        self.dataset = ComboBoxSettingCard(
                configItem = cfg.dataset,
                icon = FIF.ALBUM,
                title = self.tr("Dataset"),
                content= self.tr("Change the dataset for training"),
                texts = ["MNIST", "EMNIST-digits", "EMNIST-letters", "EMNIST-balanced", "EMNIST-byclass", "EMNIST-bymerge", "EMNIST-mnist"]
        )
        self.modelFolder = FolderListSettingCard(
            configItem = cfg.modelFolder,
            title = self.tr("Model folder"),
            content = self.tr("Change the folder where models are saved"),
            directory = rf"F:\m\MLP_learning\app\model",

        )
        
        self.setTheme = OptionsSettingCard(
            configItem = cfg.themeMode,
            icon = FIF.BRUSH,
            title = self.tr('Application theme'),
            content = self.tr("Change the appearance of your application"),
            texts = [
                self.tr('Light'), self.tr('Dark'),
                self.tr('Use system setting')
            ],
        )
        self.setZoom = OptionsSettingCard(
            configItem = cfg.dpiScale,
            icon = FIF.ZOOM,
            title = self.tr("Interface zoom"),
            content = self.tr("Change the size of widgets and fonts"),
            texts = [
                "100%", "125%", "150%", "175%", "200%",
                self.tr("Use system setting")
            ],
        )
        self.setThemeColor = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            self.tr('Theme color'),
            self.tr('Change the theme color of you application'),

        )        
        self.setMica = SwitchSettingCard(
            FIF.TRANSPARENT,
            self.tr('Mica effect'),
            self.tr('Apply semi transparent to windows and surfaces'),
            cfg.micaEnabled,
        )
        self.setLanguage = ComboBoxSettingCard(
            cfg.language,
            FIF.LANGUAGE,
            self.tr('Language'),
            self.tr('Set your preferred language for UI'),
            texts=['简体中文', 'English', self.tr('Use system setting')],
        )
        self.torchSettingsCard = SettingCardGroup(self.tr("Torch Settings"))
        self.torchSettingsCard.addSettingCards([self.useCUDA, self.dataset, self.modelFolder])
        self.sysSettingsCard = SettingCardGroup(self.tr("System Settings"))
        self.sysSettingsCard.addSettingCards([self.setMica, self.setTheme, self.setThemeColor, self.setZoom, self.setLanguage])
        self.settingLayout = QVBoxLayout()
        self.settingLayout.setSpacing(28)
        StyleSheet.SETTING_INTERFACE.apply(self)
        #self.settingLayout.setContentsMargins(36, 10, 36, 0)
        self.settingLayout.addWidget(self.torchSettingsCard, 0, Qt.AlignTop)
        self.settingLayout.addWidget(self.sysSettingsCard, 0, Qt.AlignTop)
        self.addContentLayout(self.settingLayout)
        self.__connectSignalToSlot()
        
    
    def __connectSignalToSlot(self):
        cfg.appRestartSig.connect(self.__showRestartTooltip)
        self.setMica.checkedChanged.connect(signalBus.micaEnableChanged)
        self.setTheme.optionChanged.connect(lambda : setTheme(cfg.theme))
        self.setThemeColor.colorChanged.connect(lambda c: setThemeColor(c))
        self.useCUDA.checkedChanged.connect(signalBus.deviceChanged)
        self.dataset.comboBox.currentIndexChanged.connect(signalBus.datasetChanged)
        self.setLanguage.comboBox.currentIndexChanged.connect(signalBus.languageChanged)

    def __showRestartTooltip(self):
        """ show restart tooltip """
        InfoBar.success(
            self.tr('Updated successfully'),
            self.tr('Configuration takes effect after restart'),
            duration=1500,
            parent=self
        )

        

