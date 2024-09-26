import torch

class register:
    def __init__(self, targets : list,  ):
        '''
        layer_name : hook the layer containing layer_name
        '''
        self.targets = targets
        self.layerName = ""
        self.handles = []
    def register_hook(self, module : torch.nn.Module, hook, IFprint= False):
        '''
        hook: hook function
        '''
        
        modulelist = list(module.named_children())
        for i, (name, layer) in enumerate(modulelist):
            self.layerName += "." + name
            # print(self.layerName)
            # if the layer has children, recursively register hook to them
            if len(list(layer.children())) > 0:
                self.register_hook(layer, hook, IFprint)
                pass
            flag = 0
            for target in self.targets:
                if target in self.layerName[-len(target):] and self.layerName[:len(target)] == target:
                    hookFn = layer.register_forward_hook(hook)
                    self.handles.append([str(self.layerName), hookFn])
                    flag += 1
            if IFprint:
                if flag > 0:
                    print(self.layerName, "***" * flag)
                else:
                    print(self.layerName)
            self.layerName = ".".join(self.layerName.split(".")[:-1])
            pass
        pass

    def remove_hook(self, targetHook):
        '''
        remove the targetHook in self.handles
        '''
        
        if type(targetHook) == str:
            for (i, hook) in enumerate(self.handles):
                if hook[0] == targetHook:
                    hook[1].remove()
                    self.handles.pop(i)
                    break
        elif type(targetHook) == int:
            self.handles[targetHook][1].remove()
            self.handles.pop(targetHook) 
        else:
            targetHook[1].remove()
            self.handles.remove(targetHook)
            
    def removeAll(self):
        '''
        remove the all hook in self.handles
        '''
        for targetHook in self.handles:
            if type(targetHook) == str:
                for (i, hook) in enumerate(self.handles):
                    if hook[0] == targetHook:
                        hook[1].remove()
                        self.handles.pop(i)
                        break
            elif type(targetHook) == int:
                self.handles[targetHook][1].remove()
                self.handles.pop(targetHook) 
            else:
                targetHook[1].remove()
                self.handles.remove(targetHook)

    def get_handles(self):
        return self.handles

def test_hook(module, input, output):
    # print the name and input/output shape of the layer
    str1 = module.__class__.__name__
    try:
        str2 = str(input[0].shape)
    except:
        str2 = "x"
    try:
        str3 = str(output.shape)
    except:
        str3 = str(output[0].shape)
    
    print( '%20s %30s %30s' % (str1, str2, str3))
    # FFN.append(output.clone().detach())
    pass




if __name__ == "__main__":
    from torchvision import datasets
    from Networks.model_center import get_model


    class tmp():
        def __init__(self) -> None:
            self.arch = "resnet18"
            self.activate = "relu"
            self.pretrained = False
            self.datasets = "mini_imageNet"
    tmpargs = tmp()
    model = get_model(tmpargs)
    reg = register([".layer1.0.conv2"])
    reg.register_hook(model, test_hook)
    path = "models/ResNet18_AllReLU/MiniImageNet_resnet_allReLU_199_include10%Data.ckpt"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    val_loader = torch.utils.data.DataLoader(checkpoint["test_dataset"],
        batch_size=128, shuffle=False,
        num_workers=0)
    model.to("cuda:0")
    for d, l in val_loader:
        d = d.to("cuda:0")
        l = l.to("cuda:0")
        model(d)
        pass
