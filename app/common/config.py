from PySide6.QtCore import QLocale
from qfluentwidgets import (QConfig, ConfigItem, OptionsConfigItem, BoolValidator, OptionsValidator, qconfig, ConfigSerializer,
                             FolderValidator, ConfigValidator)
from enum import Enum
import sys
import torch

CUDA_IS_AVAILABLE = torch.cuda.is_available()
def isWin11():
    return sys.platform == 'win32' and sys.getwindowsversion().build >= 22000

    

class Language(Enum):
    """ Language enumeration """

    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
    # CHINESE_TRADITIONAL = QLocale(QLocale.Chinese, QLocale.HongKong)
    ENGLISH = QLocale(QLocale.English)
    AUTO = QLocale()


class LanguageSerializer(ConfigSerializer):
    """ Language serializer """

    def serialize(self, language):
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str):
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


class MyConfig(QConfig):
    useCUDA = ConfigItem(
        "MainWindow", 
        "UseCUDA", 
        CUDA_IS_AVAILABLE, 
        BoolValidator()
        )
    dataset = OptionsConfigItem(
        "MainWindow", 
        "Dataset",
        "MNIST", 
        OptionsValidator([
            "MNIST", 
            "EMNIST-digits", "EMNIST-letters", "EMNIST-balanced", "EMNIST-byclass", "EMNIST-bymerge", "EMNIST-mnist",
            'CIFAR10'])
        )

    micaEnabled = ConfigItem("MainWindow", "MicaEnabled", isWin11(), BoolValidator())
    dpiScale = OptionsConfigItem(
        "MainWindow", "DpiScale", "Auto", OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]), restart=True)
    language = OptionsConfigItem(
        "MainWindow", "Language", Language.AUTO, OptionsValidator(Language), LanguageSerializer(), restart=True)
    modelFolder = ConfigItem("MainWindow", "ModelFolder", rf"/app/model", FolderValidator())

    model = ConfigItem("MainWindow", "Model", 'model.pt')
    # TODO: 增加没有找到任何模型时候的解决方案


cfg = MyConfig()
qconfig.load("./app/config/config.json", cfg)


# device = torch.device("cpu")
# dataType = torch.FloatTensor
