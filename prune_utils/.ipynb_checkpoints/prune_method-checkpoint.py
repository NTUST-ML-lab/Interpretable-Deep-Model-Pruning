import torch
import torch.nn as nn
from torch.nn.utils import prune

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
    
    def prune_nodes(self, module, comb, ini_node_num):
        global filters_selected
        global no_of_dimensions
        
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
            
        prune.custom_from_mask(module, name='bias', mask=bias_mask)
        
        filters_selected=[]
        no_of_dimensions=-1
        
        return module