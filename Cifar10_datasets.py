import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np

class Cifar10_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, classes=range(10), split_perc = 1., transform = None):
        if classes != "full" and isinstance(classes, list) == False:
            raise TypeError("classes's data type must be integer or value is `full`.")
        
        if not mode in {"train", "test"}:
            raise ValueError("mode content must be `train` or `test`.")
        
        istrain = True if mode=='train' else False
        
        self.classes = classes
        self.split_perc = split_perc
        self.transform = transform
        
        dataset = CIFAR10(root=f'./cifar10/{mode}', train=istrain, transform=transforms.ToTensor())
        X, Y = dataset.data, torch.tensor(dataset.targets)
        
        self.data_num = self.get_data_num(X, Y)
        Data, Label = [], []
        for cls in self.classes:
            Cifar10_data, Cifar10_label = self.class_filter(X, Y, cls, self.data_num[cls])
            Data.append(Cifar10_data)
            Label.append(Cifar10_label)
        Data = np.concatenate(Data, axis=0)
        Label = torch.cat(Label, dim=0)
        
        shuffle_index = self.shuffle(torch.arange(Data.shape[0]))
        self.data = Data[shuffle_index, :]
        Label = Label[shuffle_index]
        
        self.Natural2Label = {i:cls for i, cls in enumerate(self.classes)}
        self.Label2Natural = {cls:i for i, cls in enumerate(self.classes)}
        self.targets = torch.tensor([self.Label2Natural[l.item()] for l in Label])
        
    def get_data_num(self, data, label):
        data_num = {}
        for cls in self.classes:
            data_num[cls] = int((label==cls).sum().item() * self.split_perc)
        return data_num
    
    def shuffle(self, x):
        return x[torch.randperm(len(x))]
    
    def class_filter(self, data, label, target_class, data_num):
        idx = torch.where(label==target_class)[0]
        partial_idx = self.shuffle(idx)[:data_num]
        cls_data, cls_label = data[partial_idx, :], label[partial_idx]

        return cls_data, cls_label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.transform(self.data[idx])
        target = self.targets[idx]
        return data, target