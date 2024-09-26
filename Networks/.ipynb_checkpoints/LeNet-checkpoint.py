import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F


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
        # x = x.view(-1, 16*4*4)
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
    
class LeNet5_84_AllTanh(nn.Module):
    def __init__(self):
        super(LeNet5_84_AllTanh, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=10)

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