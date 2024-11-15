from PySide6.QtCore import Signal, QThread

class WorkerThread(QThread):
    trigger = Signal(str)
    finished = Signal()

    def __init__(self, parent=None, function=None, args=None):
        super(WorkerThread, self).__init__(parent)
        self.function = function
        self.args = args
    

    def run(self):
        # 检查函数和参数是否存在
        if self.function is not None:
            if self.args is None:
                res = self.function()
            elif isinstance(self.args, tuple):
                res = self.function(*self.args)
            elif isinstance(self.args, dict):
                res = self.function(**self.args)
            self.finished.emit()
        return res
