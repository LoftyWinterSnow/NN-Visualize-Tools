from collections import OrderedDict
from sympy import S
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
def summary(model, input_size, batch_size=-1, device="cuda", log = True):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if log:
        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if log:
            print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    try:
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    except:
        total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    if log:
        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")
    return summary

class Datasets:
    DATA_ROOT = './app/data'
    DATASETS  = {
        'MNIST': torchvision.datasets.MNIST,
        'EMNIST': torchvision.datasets.EMNIST,
        'CIFAR10': torchvision.datasets.CIFAR10,
        'CIFAR100': torchvision.datasets.CIFAR100,
        'FASHION': torchvision.datasets.FashionMNIST,
        'KMNIST': torchvision.datasets.KMNIST,
        'QMNIST': torchvision.datasets.QMNIST,
        'STL10': torchvision.datasets.STL10
    }
    TRANSFORMER = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,))
    ])
    def __init__(self, data:str, download=True):
        if 'EMNIST' in data:
            split = data.split('-')[1]
            self.trainData = Datasets.DATASETS[data.split('-')[0]](
                root=Datasets.DATA_ROOT, 
                split=split, 
                train=True, 
                transform=Datasets.TRANSFORMER, 
                download=download)
            self.testData = Datasets.DATASETS[data.split('-')[0]](
                root=Datasets.DATA_ROOT, 
                split=split, 
                train=False, 
                transform=Datasets.TRANSFORMER, 
                download=download)
        else:
            self.trainData = Datasets.DATASETS[data](
                root=Datasets.DATA_ROOT, 
                train=True, 
                transform=Datasets.TRANSFORMER, 
                download=download)
            self.testData = Datasets.DATASETS[data](
                root=Datasets.DATA_ROOT, 
                train=False, 
                transform=Datasets.TRANSFORMER, 
                download=download)
        self.name = data
        self.inputShape = tuple(self.trainData[0][0].shape)

    @property
    def classes(self):
        return self.trainData.classes

class NN:
    def __init__(self, inputShape, device):
        self.device = device
        self.model = nn.Sequential().to(self.device)
        self.inputShape = inputShape
        
    def loadModel(self, modelPth):
        self.model = torch.load(modelPth).to(self.device)
        self.modelPth = modelPth
        

    def to(self, device):
        self.model.to(device)
    

    @property
    def NNStructure(self) -> OrderedDict:
        return summary(self.model, self.inputShape, log=False)
    
    def eval(self):
        self.model.eval()

    def append(self, layer):
        self.model.append(layer)

    def lastConvLayer(self) -> int:
        convLayerIndex = -1
        structure = self.NNStructure
        
        for i, layer  in enumerate(structure.keys()):
            if "Conv" in layer:
                convLayerIndex = i
        print(convLayerIndex)
        return convLayerIndex
    
    def featureMapLayer(self) -> OrderedDict:
        layers = OrderedDict()
        structure = self.NNStructure
        for i, layer in enumerate(structure.keys()):
            if "Conv" in layer:
                layers[layer] = structure[layer]
                layers[layer]['idx'] = i
            elif "MaxPool" in layer:
                layers[layer] = structure[layer]
                layers[layer]['idx'] = i
        return layers

    @property
    def outputShape(self):
        structure = self.NNStructure
        if structure:
            return structure[list(structure.keys())[-1]]['output_shape']
        else:
            return None

    def train(self, 
              learning_rate : float, 
              batch_size : int, 
              num_epochs : int, 
              train_dataset, 
              test_dataset, 
              criterion = nn.CrossEntropyLoss(),
              teststep = 100):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # 初始化损失和准确率列表
        train_losses = []
        test_accuracies = []

        start_time = time.time()
    # 训练网络
        for epoch in range(num_epochs):
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as t:
                for i, (images, labels) in enumerate(t):
                    # 将数据和标签移动到GPU
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # 前向传播
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1) % teststep == 0:
                        # 在测试集上计算准确率
                        with torch.no_grad():
                            correct = 0
                            total = 0
                            for images, labels in test_loader:
                                # 将数据和标签移动到GPU
                                images = images.to(self.device)
                                labels = labels.to(self.device)

                    #             outputs = self.model(images)
                    #             _, predicted = torch.max(outputs.data, 1)
                    #             total += labels.size(0)
                    #             correct += (predicted == labels).sum().item()

                    #         test_accuracy = correct / total
                    #         train_losses.append(loss.item())
                    #         test_accuracies.append(test_accuracy)

                    #     # 计算训练时间
                    #     elapsed_time = time.time() - start_time

                    #     t.set_postfix_str(f'Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, TestAccuracy: {test_accuracy:.2%}, Time: {elapsed_time:.2f}s')


        return train_losses, test_accuracies

    def saveModel(self, path):
        torch.save(self.model, path)
        self.modelPth = path
    
       
    



def plot_results(train_losses, test_accuracies):
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss/Accuracy')
    plt.title('Loss and Accuracy over Steps')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
    


def get_cam_mask(model, targetLayer, image) -> np.ndarray:
    cam = GradCAM(model, [targetLayer])
    
    mask = cam(image)
    return mask


if __name__ == '__main__':
    # 加载MNIST数据集

    # train_dataset = torchvision.datasets.MNIST(root='./app/data', train=True, transform=transform, download=True)
    # test_dataset = torchvision.datasets.MNIST(root='./app/data', train=False, transform=transform, download=True)
    model = NN((3, 32, 32), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    data = Datasets('CIFAR10')
    model.model = nn.Sequential(
        nn.Conv2d(3,64,3,padding=1),
        nn.Conv2d(64,64,3,padding=1),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64,128,3,padding=1),
        nn.Conv2d(128, 128, 3,padding=1),

        nn.MaxPool2d(2, 2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128,128, 3,padding=1),
        nn.Conv2d(128, 128, 3,padding=1),
        nn.Conv2d(128, 128, 1,padding=1),
        nn.MaxPool2d(2, 2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 256, 3,padding=1),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.Conv2d(256, 256, 1, padding=1),
        nn.MaxPool2d(2, 2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.Conv2d(256, 512, 3, padding=1),
        nn.MaxPool2d(2, 2, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Flatten(),
        nn.Linear(512*3*3, 512),
        nn.ReLU(),
        nn.Linear(512, 10)

    ).to('cuda')
    #print(model.NNStructure)
    model.train(0.01, 64, 10, data.trainData, data.testData)
    # model.loadModel('./app/model/model2.pt')
    # mask = get_cam_mask(model.model, model.model[3], data.trainData[0][0].reshape((1, 1, 28, 28)))
    # print(mask)
    # model.model = nn.Sequential(
    #         nn.Conv2d(1, 32, 5, stride=1, padding=2),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Conv2d(32, 32, 5, padding=2),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Conv2d(32, 64, 5, padding=2),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Flatten(),
    #         nn.Linear(576, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 64),
    #         nn.Linear(64, 10),

    # ).to(model.device)

    # 初始化网络并移动到GPU
    # plot_results(*model.train(
    #     num_epochs=10, 
    #     batch_size=128, 
    #     learning_rate=0.001,
    #     train_dataset = train_dataset, 
    #     test_dataset = test_dataset))

    # 保存模型
    #model.saveModel("./app/model/model2.pt")
    # 加载模型
    #model = load_model("./app/model/model.pt")
    
    import code
    code.interact(local=locals())
    
    #model.get_parameter()
    #cmap = plt.get_cmap('Purples')