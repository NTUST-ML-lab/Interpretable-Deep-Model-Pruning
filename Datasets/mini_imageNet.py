


import pickle
import copy

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from torch.utils.data import Dataset
from typing import Any, Tuple




def MLclf_download_data():
    # 這個 dataset 是這樣來的
    # MLclf 他們是提供 deepAI.org 他們分的 dataset
    # 我不知道這個dataset 怎麼出來的，所以這破code不會用 就這樣

    from MLclf import MLclf
    MLclf.miniimagenet_download(Download=True)


class miniImageNetDataset(Dataset):
    def __init__(self, data, targets, classes, class_to_idx, transform = None, target_transform = None):
        '''
        data shall be (data nums, H, W, C)
        '''
        self.data = data
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)        
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def getDataset_MLclf_ver(ratio_train = 0.6, ratio_val = 0.2,
               train_transform = None,
               val_transform = None
               ):
    data_dir = "Datasets/miniimagenet/"
    dir_pickled_train = data_dir + 'mini-imagenet-cache-train.pkl'
    dir_pickled_val = data_dir + 'mini-imagenet-cache-val.pkl'
    dir_pickled_test = data_dir + 'mini-imagenet-cache-test.pkl'

    with open(dir_pickled_train, 'rb') as f:
        data_load_train = pickle.load(f)
    with open(dir_pickled_val, 'rb') as f:
        data_load_val = pickle.load(f)
    with open(dir_pickled_test, 'rb') as f:
        data_load_test = pickle.load(f)

    n_samples_train = data_load_train['image_data'].shape[0]
    n_samples_val = data_load_val['image_data'].shape[0]
    n_samples_test = data_load_test['image_data'].shape[0]

    # calculate the correct index for combine the ['class_dict']s of data_load_train, val and test.
    for i, (k, v_ls) in enumerate(data_load_val['class_dict'].items()):
        for idx, v in enumerate(v_ls):
            data_load_val['class_dict'][k][idx] = data_load_val['class_dict'][k][idx] + n_samples_train

    for i, (k, v_ls) in enumerate(data_load_test['class_dict'].items()):
        for idx, v in enumerate(v_ls):
            data_load_test['class_dict'][k][idx] = data_load_test['class_dict'][k][idx] + (n_samples_train + n_samples_val)

    data_load_all = {}
    data_load_all['image_data'] = np.concatenate((data_load_train['image_data'], data_load_val['image_data'], data_load_test['image_data']))
    data_load_all['class_dict'] = copy.deepcopy(data_load_train['class_dict'])
    data_load_all['class_dict'].update(data_load_val['class_dict'])
    data_load_all['class_dict'].update(data_load_test['class_dict'])

    key1 = list(data_load_all['class_dict'].keys())[0]
    n_samples_per_class = len(data_load_all['class_dict'][key1])
    #print('n_samples_per_class: ', n_samples_per_class) # 600
    n_class = len(list(data_load_all['class_dict'].keys()))
    #print('n_class: ', n_class)

    labels_arr_unique = np.linspace(0, n_class-1, n_class, dtype=int)
    labels_arr = np.repeat(labels_arr_unique, repeats=n_samples_per_class, axis=None)
    data_feature_label = {}

    # 原本 MLclf 是這樣寫的 我覺得不夠安全，所以處理成下面那版
    # data_feature_label['labels'] = labels_arr # 100 * 600 labels
    data_feature_label['labels'] = [-1 for _ in range( len(data_load_all['image_data']) )]
    class_to_idx = {}
    for label_idx, (key, value) in enumerate(data_load_all['class_dict'].items()):
        class_to_idx[key] = label_idx
        for image_idx in value:
            data_feature_label['labels'][image_idx] = label_idx
    for idxx, label in enumerate(data_feature_label['labels']):
        if label == -1:
            print(">>> ", idxx, "label not found")
    # =======================================================

    data_feature_label['images'] = copy.deepcopy(data_load_all['image_data']) # 100 * 600 images
    data_feature_label['labels_mark'] = list(data_load_all['class_dict'].keys()) # 100 class names.

    data_feature_label['images'] = np.array(data_feature_label['images'])
    data_feature_label['labels_mark'] = np.array(data_feature_label['labels_mark'])


    n_samples_total = len(data_feature_label['labels'])
    train_range = [0, int(np.floor(n_samples_total * ratio_train))]
    val_range = [int(np.floor(n_samples_total * ratio_train)), int(np.floor(n_samples_total * (ratio_train + ratio_val)))]
    test_range = [int(np.floor(n_samples_total * (ratio_train + ratio_val))), n_samples_total]


    if type(train_transform) == type(None):
        train_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = miniImageNetDataset( data_feature_label['images'][train_range[0] : train_range[1]],
                                        data_feature_label['labels'][train_range[0] : train_range[1]],
                                        classes=data_feature_label['labels_mark'],
                                        class_to_idx= class_to_idx,
                                        transform = train_transform
                                          )
    
    if type(val_transform) == type(None):
        val_transform = transforms.Compose([transforms.ToTensor()])

    val_dataset = miniImageNetDataset( data_feature_label['images'][val_range[0] : val_range[1]],
                                        data_feature_label['labels'][val_range[0] : val_range[1]],
                                        classes=data_feature_label['labels_mark'],
                                        class_to_idx= class_to_idx,
                                        transform = val_transform
                                          )
    test_dataset = miniImageNetDataset( data_feature_label['images'][test_range[0] : test_range[1]],
                                        data_feature_label['labels'][test_range[0] : test_range[1]],
                                        classes=data_feature_label['labels_mark'],
                                        class_to_idx= class_to_idx,
                                        transform = val_transform
                                          )


    return train_dataset, val_dataset, test_dataset
