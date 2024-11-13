# coding:utf-8
from PySide6.QtCore import Qt, QUrl, QEvent, QThread
from PySide6.QtGui import QDesktopServices, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QFormLayout
from qfluentwidgets import (ScrollArea, LineEdit, ToolButton, FluentIcon, Action, MessageBox,Flyout,FlyoutAnimationType,
                            isDarkTheme, ComboBox, Theme, ToolTipFilter, TitleLabel, CaptionLabel,EditableComboBox,
                            StrongBodyLabel, MessageBoxBase, toggleTheme, SubtitleLabel, IndeterminateProgressBar, CommandBar,
                            qconfig, PushButton)
from ..common.style_sheet import StyleSheet


class WidgetCard(QWidget):
    """ Widget card """

    def __init__(self, title, widgets: list ,topLayout = "H", stretch=0, gap = 10, parent=None):
        super().__init__(parent=parent)
        self.widgets = widgets
        self.stretch = stretch

        self.titleLabel = StrongBodyLabel(title, self)
        self.titleLayout = QVBoxLayout()
        self.card = QFrame(self)
        self.layoutDir = topLayout
        self.vBoxLayout = QVBoxLayout(self)
        self.cardLayout = QVBoxLayout(self.card)
        if topLayout == "H":
            self.topLayout = QHBoxLayout()
        else:
            self.topLayout = QVBoxLayout()
        self.gap = gap
        
        self.__initWidget()

    def __initWidget(self):
        self.__initLayout()
        self.card.setObjectName('card')


    def __initLayout(self):
        self.vBoxLayout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        self.cardLayout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        self.topLayout.setSizeConstraint(QHBoxLayout.SetMinimumSize)
        self.topLayout.setSpacing(self.gap)
        self.vBoxLayout.setSpacing(10)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.topLayout.setContentsMargins(10, 10, 10, 10)
        self.cardLayout.setContentsMargins(0, 0, 0, 0)

        self.titleLayout.addWidget(self.titleLabel, 0, Qt.AlignLeft)
        #self.titleLayout.addStretch(1)
        self.vBoxLayout.addLayout(self.titleLayout, 0)
        self.vBoxLayout.addWidget(self.card, 0, Qt.AlignTop)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

        self.cardLayout.setSpacing(0)
        self.cardLayout.setAlignment(Qt.AlignTop)
        self.cardLayout.addLayout(self.topLayout, 0)

        for widget in self.widgets:
            if isinstance(widget, list):
                tempLayout = QVBoxLayout() if self.layoutDir == "H" else QHBoxLayout()
                tempLayout.setSpacing(self.gap)
                for groupWiget in widget:
                    tempLayout.addWidget(groupWiget)
                self.topLayout.addLayout(tempLayout)
            else:
                widget.setParent(self.card)
                self.topLayout.addWidget(widget)

        if self.stretch == 0:
            self.topLayout.addStretch(1)

        #self.widget.show()


    def addWidgets(self, widgets):
        self.widgets = widgets
        for widget in self.widgets:
            if isinstance(widget, list):
                tempLayout = QVBoxLayout() if self.layoutDir == "H" else QHBoxLayout()
                tempLayout.setSpacing(self.gap)
                for groupWiget in widget:
                    tempLayout.addWidget(groupWiget)
                self.topLayout.addLayout(tempLayout)
            else:
                widget.setParent(self.card)
                self.topLayout.addWidget(widget)

        self.update()
    
    def addTools(self, widget):
        self.topLayout.addWidget(widget)

    def clear(self):
        while self.topLayout.count():
            item = self.topLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

class SeparatorWidget(QWidget):
    """ Seperator widget """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(6, 16)

    def paintEvent(self, e):
        painter = QPainter(self)
        pen = QPen(1)
        pen.setCosmetic(True)
        c = QColor(255, 255, 255, 21) if isDarkTheme() else QColor(0, 0, 0, 15)
        pen.setColor(c)
        painter.setPen(pen)

        x = self.width() // 2
        painter.drawLine(x, 0, x, self.height())

class ToolBar(QWidget):
    """ Tool bar """
    def __init__(self, title, subtitle,parent=None):
        super().__init__(parent=parent)
        self.titleLabel = TitleLabel(title, self)
        self.subtitleLabel = CaptionLabel(subtitle, self)
        self.themeButton = ToolButton(FluentIcon.CONSTRACT, self)
        self.vBoxLayout = QVBoxLayout(self)
        self.buttonLayout = QHBoxLayout()

        self.titleLayout = QVBoxLayout()
        self.costumBtnLayout = QHBoxLayout()
        self.__initWidget()

    def __initWidget(self):

        self.subtitleLabel.setTextColor(QColor(96, 96, 96), QColor(216, 216, 216))
        
        self.themeButton.installEventFilter(ToolTipFilter(self.themeButton))
        self.themeButton.setToolTip(self.tr('Toggle theme'))
        self.themeButton.clicked.connect(lambda: toggleTheme(True))

        self.titleLayout.setSpacing(0)
        self.titleLayout.setContentsMargins(0, 0, 0, 0)
        self.titleLayout.addWidget(self.titleLabel)
        self.titleLayout.setSpacing(4)
        self.titleLayout.addWidget(self.subtitleLabel)
        self.titleLayout.setSpacing(4)

        self.setFixedHeight(120)

        self.vBoxLayout.setContentsMargins(36, 22, 36, 12)
        
        self.vBoxLayout.addLayout(self.titleLayout, 0)
        self.vBoxLayout.addLayout(self.buttonLayout, 1)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

        self.buttonLayout.setSpacing(4)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)

        self.buttonLayout.addLayout(self.costumBtnLayout, 0)
        self.buttonLayout.addStretch(1)
        self.buttonLayout.addWidget(self.themeButton, 0, Qt.AlignRight)
        self.buttonLayout.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    
    def addTools(self, tools):
        for tool in tools:
            self.costumBtnLayout.addWidget(tool, 0, Qt.AlignLeft)
        
        

class Content(ScrollArea):
    """ Content """

    def __init__(self, title: str, subtitle : str , showToolBar = True, parent=None):
        """
        Parameters
        ----------
        title: str
            The title of gallery

        subtitle: str
            The subtitle of gallery

        parent: QWidget
            parent widget
        """
        super().__init__(parent=parent)
        self.view = QWidget(self)
        if showToolBar:
            self.toolBar = ToolBar(title, subtitle, self)
            self.setViewportMargins(0, self.toolBar.height(), 0, 0)
        else:
            self.toolBar = None
            self.setViewportMargins(0, 0, 0, 0)
            
        self.contentLayout = QVBoxLayout(self.view)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.contentLayout.setSpacing(10)
        self.contentLayout.setAlignment(Qt.AlignTop)
        if showToolBar:
            self.contentLayout.setContentsMargins(36, 0, 36, 36)
        else:
            self.contentLayout.setContentsMargins(36, 36, 36, 36)

        self.setObjectName(title)
        self.view.setObjectName('view')
        StyleSheet.CONTENT_INTERFACE.apply(self)

    def addContentLayout(self, layout):
        self.contentLayout.addLayout(layout, 0)

    def showError(self, title, content):
        self.errMessage = MessageBox(title, content, self)
        if self.errMessage.exec():
            print('Yes button is pressed')
        else:
            print('Cancel button is pressed')
        
    def setSubTitle(self, title):
        if self.toolBar != None:
            self.toolBar.subtitleLabel.setText(title)

            

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.toolBar != None:
            self.toolBar.resize(self.width(), self.toolBar.height())
