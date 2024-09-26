import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor

class MNIST_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, classes=range(10), split_perc = 1.):
        if classes != "full" and isinstance(classes, list) == False:
            raise TypeError("classes's data type must be integer or value is `full`.")
        
        if not mode in {"train", "test"}:
            raise ValueError("mode content must be `train` or `test`.")
        
        istrain = True if mode=='train' else False
        
        self.classes = classes
        self.split_perc = split_perc
        self.transform = ToTensor()
        
        dataset = mnist.MNIST(root=f'./mnist/{mode}', train=istrain, transform=self.transform)
        MNIST_X, MNIST_Y = dataset.data, dataset.targets
        
        self.data_num = self.get_data_num(MNIST_X, MNIST_Y)
        Data, Label = [], []
        for cls in self.classes:
            MNIST_data, MNIST_label = self.class_filter(MNIST_X, MNIST_Y, cls, self.data_num[cls])
            Data.append(MNIST_data)
            Label.append(MNIST_label)
        Data = torch.cat(Data, dim=0)
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
        data = torch.unsqueeze(self.data[idx], dim=0)/255
        target = self.targets[idx]
        return data, target
    
class MNIST_Dataset_32(torch.utils.data.Dataset):
    def __init__(self, mode, classes=range(10), split_perc = 1.):
        if classes != "full" and isinstance(classes, list) == False:
            raise TypeError("classes's data type must be integer or value is `full`.")
        
        if not mode in {"train", "test"}:
            raise ValueError("mode content must be `train` or `test`.")
        
        istrain = True if mode=='train' else False
        
        self.classes = classes
        self.split_perc = split_perc
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize([32, 32]),
                                             transforms.ToTensor(),
                                            ])
        
        dataset = mnist.MNIST(root=f'./mnist/{mode}', train=istrain, transform=transforms.ToTensor())
        MNIST_X, MNIST_Y = dataset.data, dataset.targets
        
        self.data_num = self.get_data_num(MNIST_X, MNIST_Y)
        Data, Label = [], []
        for cls in self.classes:
            MNIST_data, MNIST_label = self.class_filter(MNIST_X, MNIST_Y, cls, self.data_num[cls])
            Data.append(MNIST_data)
            Label.append(MNIST_label)
        Data = torch.cat(Data, dim=0)
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