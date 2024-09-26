import torch
import torch.nn as nn
from torch.nn.utils import prune
import numpy as np
from copy import deepcopy

class PruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    dim=0
    def compute_mask(self, t, default_mask):
        global filters_selected
        global no_of_dimensions
        
        mask = default_mask.clone()
        #print("the mask size is ",mask.size())
        #print(self.pruned_filters["conv2"])
        if(no_of_dimensions==4):
            for i,val in enumerate(filters_selected):
                if(val==0):
                    mask[i,:,:,:]=0
        if(no_of_dimensions==2):
            for i,val in enumerate(filters_selected):
                if(val==0):
                    mask[i,:]=0
        if(no_of_dimensions==1):
            for i,val in enumerate(filters_selected):
                if(val==0):
                    mask[i]=0  
        return mask
    
    
    def prune_nodes(self, module, comb, ini_node_num, remove = False):
        global filters_selected
        global no_of_dimensions
        
        if remove:
            module.out_channels = len(comb)
            module = module.weight[comb, :]
            return module

        filters_selected = [1 if node in comb else 0 for node in range(ini_node_num)]

        #----------------pruning of conv layer weight and bias----------
        #print(module)
        if(isinstance(module, torch.nn.Conv2d)):
            no_of_dimensions=4
            PruningMethod.apply(module,"weight")
            bias_mask = (torch.sum(module.weight_mask, axis=(1, 2, 3)) != 0).to(torch.float32)

        #----------------pruning of FC layer weight and bias----------
        else:
            no_of_dimensions=2
            PruningMethod.apply(module,"weight")
            bias_mask = (torch.sum(module.weight_mask, axis=(1)) != 0).to(torch.float32)
        if module.bias is not None:
            prune.custom_from_mask(module, name='bias', mask=bias_mask)
        
        filters_selected=[]
        no_of_dimensions=-1
        
        return module


class simMask():
    def __init__(self, node_num) -> None:
        self.node_dict = {i:i for i in range(node_num)}
        self.node_num = node_num

    def reset(self):
        self.node_dict = {i:i for i in range(self.node_num)}

    def remove(self, idx_list):
        '''
        0:0 1:-1 2:1 3:2 4:3 5:4
        rm 125

        step 1
        removeAble_idx = 24
        rm_idx_trans = [1, 3]
        0:0 1:-1 2:-1 3:2 4:-1 5:4

        step 2
        0:0 1:-1 2:-1 3:1 4:-1 5:2
        '''
        
        removeAble_idx = [idx for idx in idx_list if self.node_dict[idx] >= 0]
        rm_idx_trans = []
        # step 1
        for idx in removeAble_idx:
            rm_idx_trans.append(self.node_dict[idx])
            self.node_dict[idx] = -1
            
        # step 2
        new_idx = 0
        for i in range(self.node_num):
            if self.node_dict[i] != -1:
                self.node_dict[i] = new_idx
                new_idx += 1
        return rm_idx_trans

    def retain(self, idx_list):
        sett = set()
        for idx in idx_list:
            sett.add(idx)
        old_dict = deepcopy(self.node_dict)
        remove_idx = [i for i in range(self.node_num) if i not in sett]
        self.remove(remove_idx)

        retain_idx = []
        for i in range(self.node_num):
            if self.node_dict[i] != -1 and old_dict[i] != -1:
                retain_idx.append(old_dict[i])

        return retain_idx
    
    def get_remain_node(self):
        return sum([1 for value in self.node_dict.values() if value >= 0])




def pruning(model, layer_name, node_num, comb, ifPrint = False):
    PMethod = PruningMethod()
    for name, layer_module in model.named_modules():
        if(isinstance(layer_module, torch.nn.Linear) and name == layer_name):
            if ifPrint:
                print(name, layer_module)
            PMethod.prune_nodes(layer_module, comb, node_num)
        if(isinstance(layer_module, torch.nn.Conv2d) and name == layer_name):
            if ifPrint:
                print(name, layer_module)
            PMethod.prune_nodes(layer_module, comb, node_num)
    return model