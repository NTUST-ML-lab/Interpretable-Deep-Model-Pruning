import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models_imagenet

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

import Datasets.mini_imageNet as mini_imageNet
from Datasets import  transforms

class sampleArgs():
    def __init__(self) -> None:
        self.datasets = "mini_imageNet"
        self.transform = "timm"
        self.batch_size = 128
        self.workers = 1



def get_transform(args):
    if args.transform.lower() == "CEC".lower():
        return {'train_transform':None, 'test_transform': None}
    if args.transform.lower() == "timm".lower():
        import timm
        return transforms.get_timm_transform(timm.create_model("resnet18"))
    if args.transform.lower() == "dino".lower():
        return transforms.get_dino_transform()
    if args.transform.lower() == "resnet".lower():
        return transforms.get_resnet_transform()

def get_datasets(args, val = False, onlyTrain = False):
    if args == None:
        args = sampleArgs()


    if args.datasets == 'ImageNet':
        traindir = os.path.join('D:/Datasets/imageNet/ILSVRC2012', 'train')
        valdir = os.path.join('D:/Datasets/imageNet/ILSVRC2012', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)
    

    elif args.datasets == 'mini-imageNet_MLclf_ver':
        # 已棄用

        # Transform 是從這篇幹來的：
        #     @inproceedings{ahmed2024orco,
        #         title={OrCo: Towards Better Generalization via Orthogonality and Contrast for Few-Shot Class-Incremental Learning},
        #         author={Ahmed, Noor and Kukleva, Anna and Schiele, Bernt},
        #         booktitle={41st IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        #         year={2024},
        #         organization={IEEE}
        #     }
        # https://github.com/noorahmedds/OrCo/blob/main/dataloader/miniimagenet/miniimagenet.py
        
        # 本來想從這篇來 不過後面總覺得怪怪 就算了 
        # Hu, Shell Xu, et al. "Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

        image_size = 84
        train_Transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),     # TODO: originally was toggled off
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        val_transform = transforms.Compose([
            transforms.Resize([92, 92]),                # TODO: Why the choice of 92 as the resize parameter?
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

        train_dataset, val_loader, test_dataset = mini_imageNet.getDataset_MLclf_ver(ratio_train=0.6, ratio_val=0.2, train_transform=train_Transform, val_transform=val_transform)

    elif args.datasets == "mini_imageNet":
        # 詳情見 Datasets.CEC_miniimagenet

        from Datasets import CEC_miniimagenet, transforms
        
        transform = get_transform(args)
        train_set = CEC_miniimagenet.MiniImageNet(train=True, transform=transform['train_transform'])
        test_set = CEC_miniimagenet.MiniImageNet(train=False, transform=transform['test_transform'])

        if not val:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
    
            val_loader = torch.utils.data.DataLoader(test_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers)
            
            if not onlyTrain:
                return train_loader, val_loader, test_set
            return train_loader
        else:
            import random 
            from torch.utils.data import Subset
            random.seed(42)
            
            train_indices = []
            for idx, (_, label) in enumerate(train_set):
                train_indices.append(idx)
            random.shuffle(train_indices)
            split_index = int(len(train_indices) * 0.8)
            val_indices = train_indices[split_index:]
            train_indices = train_indices[:split_index]
            train_dataset = Subset(train_set, train_indices)


            val_set = CEC_miniimagenet.MiniImageNet(train=True, transform=transform['test_transform'])
            val_dataset = Subset(val_set, val_indices)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            
            val_loader = torch.utils.data.DataLoader(val_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers)
            
            test_loader = torch.utils.data.DataLoader(test_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers)
            
            if not onlyTrain:
                return train_loader, val_loader, test_loader, val_dataset, test_set
            else:
                return train_loader, val_loader, val_dataset
    






