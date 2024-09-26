import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
from Networks.network import Network

# class Model(Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(256, 120)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(120, 84)
#         self.relu4 = nn.ReLU()
#         self.fc3 = nn.Linear(84, 10)
#         self.relu5 = nn.ReLU()

#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.relu1(y)
#         y = self.pool1(y)
#         y = self.conv2(y)
#         y = self.relu2(y)
#         y = self.pool2(y)
#         y = y.view(y.shape[0], -1)
#         y = self.fc1(y)
#         y = self.relu3(y)
#         y = self.fc2(y)
#         y = self.relu4(y)
#         y = self.fc3(y)
#         y = self.relu5(y)
#         return y



class LeNet5(nn.Module, Network):
    def __init__(self, classes = 10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, classes)
        self.a_type='relu'
        for m in self.modules():
            self.weight_init(m)
        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, x):
        conv1_out =self.conv1(x)
        conv1_act = F.relu(conv1_out)
        layer1 = F.max_pool2d( conv1_act, (2, 2))
        conv2_out = self.conv2(layer1)
        conv2_act = F.relu(conv2_out)
        layer2 = F.max_pool2d(conv2_act, 2)
        layer2_p = layer2.view(-1, int(layer2.nelement() / layer2.shape[0]))
        fc1_out = self.fc1(layer2_p)
        layer3= F.relu(fc1_out)
        fc2_out = self.fc2(layer3)
        layer4 = F.relu(fc2_out)
        layer5 = self.fc3(layer4)
        return layer5, [conv1_act, conv2_act, fc1_out, fc2_out]

class LeNet5_128_64_tanh(nn.Module):
    def __init__(self, classes = 10):
        super(LeNet5_128_64_tanh, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, classes)
        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, x):
        conv1_out =self.conv1(x)
        conv1_act = torch.tanh(conv1_out)
        layer1 = F.max_pool2d( conv1_act, (2, 2))
        conv2_out = self.conv2(layer1)
        conv2_act = torch.tanh(conv2_out)
        layer2 = F.max_pool2d(conv2_act, 2)
        layer2_p = layer2.view(-1, int(layer2.nelement() / layer2.shape[0]))
        fc1_out = self.fc1(layer2_p)
        layer3= torch.tanh(fc1_out)
        fc2_out = self.fc2(layer3)
        layer4 = torch.tanh(fc2_out)
        layer5 = self.fc3(layer4)
        return layer5, [conv1_out, conv2_out, fc1_out, fc2_out]






class LeNet(Module):
    def __init__(self, node_num):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, node_num)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(node_num, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        z = self.relu4(y)
        y = self.fc3(z)
        y = self.relu5(y)
        return y, z
    
class LeNet_tanh(Module):
    def __init__(self, node_num):
        super(LeNet_tanh, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, node_num)
        self.relu4 = nn.Tanh()
        self.fc3 = nn.Linear(node_num, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        z = self.relu4(y)
        y = self.fc3(z)
        y = self.relu5(y)
        return y, z

    
class LeNet_300_100_Tanh(nn.Module):
    def __init__(self):
        super(LeNet_300_100_Tanh, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        z = F.tanh(self.fc2(x))
        x = self.fc3(z)
        return x, z
    
class LeNet5_16_Tanh(nn.Module):
    def __init__(self):
        super(LeNet5_16_Tanh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        z = F.tanh(self.fc2(x))
        x = self.fc3(z)
        return x, z
    
class LeNet5_84_Tanh(nn.Module):
    def __init__(self):
        super(LeNet5_84_Tanh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        z = F.tanh(self.fc2(x))
        x = self.fc3(z)
        return x, z
    
################################################################################
# AllTanh
################################################################################
class LeNet_120_16_Tanh(nn.Module):
    def __init__(self, classnum):
        super(LeNet_120_16_Tanh, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=120)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=16)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=16, out_features=classnum)

    def forward(self, x):
        x = torch.flatten(x, 1)
        z1 = self.tanh1 (self.fc1(x))
        z2 = self.tanh2(self.fc2(z1))
        x = self.fc3(z2)
        return x, [z1, z2]

class LeNet5_84_AllTanh(nn.Module):
    def __init__(self, class_num):
        super(LeNet5_84_AllTanh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=class_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        z1 = self.tanh1(self.fc1(x))
        z2 = self.tanh2(self.fc2(z1))
        x = self.fc3(z2)
        return x, z1, z2
    
class LeNet6_16_AllTanh(nn.Module):
    def __init__(self, class_num):
        super(LeNet6_16_AllTanh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=16)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(in_features=16, out_features=class_num)

    def forward(self, x):
        c1 = self.conv1(x)
        c1_act = F.relu(c1)
        x = F.avg_pool2d(c1_act, kernel_size=2, stride=2)
        c2 = self.conv2(x)
        c2_act = F.relu(c2)
        x = F.avg_pool2d(c2_act, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        z1 = self.tanh1(self.fc1(x))
        z2 = self.tanh2(self.fc2(z1))
        z3 = self.tanh3(self.fc3(z2))
        x = self.fc4(z3)
        return x, [z1, z2, z3, c1_act, c2_act]
    
    def get_layer(self):
        return ["conv1", "conv2", "fc1", "fc2", "fc3"]

class LeNet6_16_AllTanh_bf_act(nn.Module):
    def __init__(self, class_num):
        super(LeNet6_16_AllTanh_bf_act, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=16)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(in_features=16, out_features=class_num)

    def forward(self, x):
        c1 = self.conv1(x)
        c1_act = F.relu(c1)
        x = F.avg_pool2d(c1_act, kernel_size=2, stride=2)
        c2 = self.conv2(x)
        c2_act = F.relu(c2)
        x = F.avg_pool2d(c2_act, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        z1 = self.tanh1(self.fc1(x))
        z2 = self.tanh2(self.fc2(z1))
        z3 = self.tanh3(self.fc3(z2))
        x = self.fc4(z3)
        return x, [z1, z2, z3, c1, c2]
    
    def get_layer(self):
        return ["conv1", "conv2", "fc1", "fc2", "fc3"]
    
class LeNet7_16_AllTanh(nn.Module):
    def __init__(self, class_num):
        super(LeNet7_16_AllTanh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=64)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(in_features=64, out_features=16)
        self.tanh4 = nn.Tanh()
        self.fc5 = nn.Linear(in_features=16, out_features=class_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        z1 = self.tanh1(self.fc1(x))
        z2 = self.tanh2(self.fc2(z1))
        z3 = self.tanh3(self.fc3(z2))
        z4 = self.tanh4(self.fc4(z3))
        x = self.fc5(z4)
        return x, [z1, z2, z3, z4]
    
class LeNet8_16_AllTanh(nn.Module):
    def __init__(self, class_num):
        super(LeNet8_16_AllTanh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=64)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.tanh4 = nn.Tanh()
        self.fc5 = nn.Linear(in_features=32, out_features=16)
        self.tanh5 = nn.Tanh()
        self.fc6 = nn.Linear(in_features=16, out_features=class_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        z1 = self.tanh1(self.fc1(x))
        z2 = self.tanh2(self.fc2(z1))
        z3 = self.tanh3(self.fc3(z2))
        z4 = self.tanh4(self.fc4(z3))
        z5 = self.tanh5(self.fc5(z4))
        x = self.fc6(z5)
        return x, [z1, z2, z3, z4, z5]
    
class LeNet9_16_AllTanh(nn.Module):
    def __init__(self, class_num):
        super(LeNet9_16_AllTanh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=64)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.tanh4 = nn.Tanh()
        self.fc5 = nn.Linear(in_features=32, out_features=16)
        self.tanh5 = nn.Tanh()
        self.fc6 = nn.Linear(in_features=16, out_features=16)
        self.tanh6 = nn.Tanh()
        self.fc7 = nn.Linear(in_features=16, out_features=class_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        z1 = self.tanh1(self.fc1(x))
        z2 = self.tanh2(self.fc2(z1))
        z3 = self.tanh3(self.fc3(z2))
        z4 = self.tanh4(self.fc4(z3))
        z5 = self.tanh5(self.fc5(z4))
        z6 = self.tanh6(self.fc6(z5))
        x = self.fc7(z6)
        return x, [z1, z2, z3, z4, z5, z6]