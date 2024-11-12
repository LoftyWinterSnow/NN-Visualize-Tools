# coding: utf-8
from PySide6.QtCore import QObject, Signal


class SignalBus(QObject):
    """ Signal bus """

    switchToSampleCard = Signal(str, int)
    micaEnableChanged = Signal(bool)
    deviceChanged = Signal()
    supportSignal = Signal()
    datasetChanged = Signal()
    canvasUpdate = Signal()
    languageChanged = Signal()
    


signalBus = SignalBus()