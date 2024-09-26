from functools import partial
from Networks.LeNet import LeNet6_16_AllTanh
from pytorch_util import count_parameters, count_target_ops
import torch
from copy import deepcopy
import torch_pruning as tp
import resnet18_prune
cand = []

def calFLOPs(modules):
    params = 0
    for module in modules:
        for item in module:
            params += item[0].numel()
    return params




def __(model, targetFLOPs):
    example_input = torch.randn( (1, 1, 28, 28) )

    a = count_target_ops(model, example_input)
    b = count_parameters(model)

    modules = {}
    layerName = ""
    origin_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        layerName = name.split(".")[0]
        origin_params += parameter.numel()
        if len(parameter.shape) == 4:
            module = [parameter, parameter.shape[1], parameter.shape[0] ]
        elif len(parameter.shape) == 2:
            module = [parameter, parameter.shape[1], parameter.shape[0] ]
        elif len(parameter.shape) == 1:
            module = [parameter, parameter.shape[0]]
        else:
            raise Exception("this module is not defined")
        if layerName not in modules:
            modules[layerName] = {"nodes" : parameter.shape[0], "module" :[]}

        modules[layerName]["module"].append(module)
        if "bias" in name:
            layerParam = parameter.numel() + modules[layerName]["module"][0][0].numel()
            layerParam /= parameter.shape[0]
    return modules

def depgraph_pruning(ori_model:torch.nn.Module, TargetRelativeFLOPs: float, example_inputs = torch.randn((1, 1, 28, 28)), classes = 10, 
                     device = "cuda:0", FConly = False, iter_num = 2000, bigger = False, L1_style = False):
    model = deepcopy(ori_model)
    model.to(device)
    example_inputs = example_inputs.to(device)
    imp = tp.importance.GroupNormImportance(p=2)
    pruner_entry = partial(tp.pruner.GroupNormPruner, reg=5e-4, global_pruning=False)
    ignored_layers = []
    unwrapped_parameters = []
    ch_sparsity_dict = {}
    if not FConly:
        for m, (name, m) in enumerate(model.named_modules()):
            if isinstance(m, torch.nn.Linear) and m.out_features == classes:
                ignored_layers.append(m) 
            if L1_style:
                if isinstance(m, torch.nn.Conv2d) and name in ["conv1", "layer1.0.conv2", "layer1.1.conv2"]:
                    ignored_layers.append(m) 
    else:
        for idx, (name, m) in enumerate(model.named_modules()):
            if idx == 0:
                continue
            if isinstance(m, torch.nn.Linear) and m.out_features == classes:
                ignored_layers.append(m) 
            if not isinstance(m, torch.nn.Linear):
                ignored_layers.append(m) 


    if FConly:
        targetLayer = ["fc1", "fc2", "fc3", "fc4"]
    else:
        targetLayer = []
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iter_num,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        # max_ch_sparsity=1.0,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    model.eval()
    closerest_model = ""
    ori_parameters = count_target_ops(model, example_inputs, targets=targetLayer)
    closerest_remain = {}
    closerest = 1e10
    for i in range(iter_num):
        pruner.step()
        RF =  count_target_ops(model, example_inputs, targets=targetLayer)[1] / ori_parameters[1]
        # if bigger and RF < TargetRelativeFLOPs:
        #     break
        if abs(TargetRelativeFLOPs - RF) < closerest:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    closerest_remain[name] = module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    closerest_remain[name] = module.out_channels
            closerest = abs(TargetRelativeFLOPs - RF)
            closerest_RF = RF
            closerest_model = deepcopy(model)
        if RF < TargetRelativeFLOPs:
            break
    return closerest_model, closerest_remain, closerest_RF


def calRF_node_L1(ori_model:torch.nn.Module, TargetRelativeFLOPs: float,
                  example_inputs = torch.randn((1, 1, 28, 28)), device = "cuda:0",
                  iter_num = 2000, rf = False):
    model = deepcopy(ori_model)
    model.to(device)
    example_inputs = example_inputs.to(device)
    ori_parameters = count_target_ops(model, example_inputs)
    idxx = 1 if not rf else 0 
    closerest_RF = 0
    closerest_model = deepcopy(model.cpu())
    closerest = 1e10
    closerest_remain = {}
    for iter in range(iter_num-1, 0, -1):
        model = deepcopy(ori_model)
        model = resnet18_prune.prune_resnet_L1(model, iter / iter_num, device=device)
        RF =  count_target_ops(model, example_inputs)[idxx] / ori_parameters[idxx]
        if abs(TargetRelativeFLOPs - RF) < closerest:
            closerest_RF = abs(TargetRelativeFLOPs - RF)
            closerest_model = deepcopy(model)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    closerest_remain[name] = module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    closerest_remain[name] = module.out_channels

        if RF < TargetRelativeFLOPs:
            break
    return closerest_model, closerest_remain


def calRF_node(ori_model:torch.nn.Module, TargetRelativeFLOPs: float, 
               example_inputs = torch.randn((1, 1, 28, 28)), classes = 10, 
               device = "cuda:0", iter_num = 2000, rf = False):
    model = deepcopy(ori_model)
    model.to(device)
    example_inputs = example_inputs.to(device)
    imp = tp.importance.RandomImportance()
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == classes:
            ignored_layers.append(m) 


    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        iterative_steps = iter_num,
        pruning_ratio=0.9999, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
    )
    ori_parameters = count_target_ops(model, example_inputs)
    closerest = 1e10
    closerest_RF = 0
    closerest_remain = {}
    idxx = 1 if not rf else 0 
    for i in range(iter_num):
        pruner.step()
        RF =  count_target_ops(model, example_inputs)[idxx] / ori_parameters[idxx]
        if abs(TargetRelativeFLOPs - RF) < closerest:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    closerest_remain[name] = module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    closerest_remain[name] = module.out_channels
            closerest = abs(TargetRelativeFLOPs - RF)
            closerest_RF = RF
            pass
            if RF < TargetRelativeFLOPs:
                break
    return closerest_remain
    
def calRF_FC(ori_model:torch.nn.Module, TargetRelativeFLOPs: float, example_inputs = torch.randn((1, 1, 28, 28)), classes = 10, device = "cuda:0"):
    model = deepcopy(ori_model)
    model.to(device)
    example_inputs = example_inputs.to(device)
    imp = tp.importance.RandomImportance()
    ignored_layers = []
    for idx, m in enumerate(model.modules()):
        if idx == 0:
            continue
        if isinstance(m, torch.nn.Linear) and m.out_features == classes:
            ignored_layers.append(m) 
        if not isinstance(m, torch.nn.Linear):
            ignored_layers.append(m) 

    iter_num =2000

    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        iterative_steps = iter_num,
        pruning_ratio=0.9999, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
    )
    ori_parameters = count_target_ops(model, example_inputs, targets=["fc1", "fc2", "fc3", "fc4"])
    closerest = 1e10
    closerest_RF = 0
    closerest_remain = {}
    for i in range(iter_num):
        pruner.step()
        RF =  count_target_ops(model, example_inputs, targets=["fc1", "fc2", "fc3", "fc4"])[1] / ori_parameters[1]
        if abs(TargetRelativeFLOPs - RF) < closerest:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    closerest_remain[name] = module.out_features
            closerest = abs(TargetRelativeFLOPs - RF)
            closerest_RF = RF
            pass
    return closerest_remain



def lenetPruneSim(classNum, conv1, conv2, fc1, fc2, fc3):
    model = LeNet6_16_AllTanh(classNum)
    
    conv1RM = [i for i in range(0, model.conv1.out_channels - conv1)]
    tp.prune_conv_out_channels(model.conv1, conv1RM )
    tp.prune_conv_in_channels(model.conv2, conv1RM )

    conv2RM = [i for i in range(0, model.conv2.out_channels - conv2)]
    fc1_inRM = [i for i in range(0, model.fc1.in_features - conv2 *4 *4)]
    tp.prune_conv_out_channels(model.conv2, conv2RM )
    tp.prune_linear_in_channels(model.fc1, fc1_inRM )

    fc1RM = [i for i in range(0, model.fc1.out_features - fc1)]
    tp.prune_linear_out_channels(model.fc1, fc1RM )
    tp.prune_linear_in_channels(model.fc2, fc1RM )

    fc2RM = [i for i in range(0, model.fc2.out_features - fc2)]
    tp.prune_linear_out_channels(model.fc2, fc2RM )
    tp.prune_linear_in_channels(model.fc3, fc2RM )

    fc3RM = [i for i in range(0, model.fc3.out_features - fc3)]
    tp.prune_linear_out_channels(model.fc3, fc3RM )
    tp.prune_linear_in_channels(model.fc4, fc3RM )

    return model

# sim(LeNet6_16_AllTanh(10), 0.9)

if __name__ == "__main__":
    import torch_pruning as tp
    from Networks.model_center import get_model
    from pytorch_util import count_target_ops
    from thop import profile
    from torchstat import stat
    from resnet18_prune import resnet18Pruner

    CLASS = 10
    model_ori = LeNet6_16_AllTanh(CLASS)
    a = lenetPruneSim(CLASS,1, 16, 30, 21, 4)
    stat( a, (1, 28, 28))
    orim, orip = count_target_ops(model_ori, torch.randn((1, 1, 28, 28)))
    am, ap = count_target_ops(a, torch.randn((1, 1, 28, 28)))
    pass



    # class tmp():
    #     def __init__(self) -> None:
    #         self.arch = "resnet18"
    #         self.activate = "relu"
    #         self.pretrained = False
    #         self.datasets = "mini_imageNet"
    
    # def getRange(rangInt):
    #     return list(range(i for i in range(rangInt)))

    # model = get_model(tmp())
    # stat( model, (3, 224, 224))
    # PM = calRF_node_L1(ori_model=model, TargetRelativeFLOPs=0.8, 
    #                    example_inputs= torch.randn((1, 3, 224, 224)), iter_num=200)
    
    # stat( PM, (3, 224, 224))
    # pass



    # rp.prune(model, layer, removed_comb)

    # class tmp():
    #     def __init__(self) -> None:
    #         self.arch = "resnet18"
    #         self.activate = "relu"
    #         self.pretrained = False
    #         self.datasets = "mini_imageNet"
    # model = get_model(tmp())
    
    # a = depgraph_pruning(model, 0.1, torch.randn(1, 3, 80, 80), 100)
    # print(a)

    pass


    # calRF_node(model, 0.9)
    # pass
    # example_inputs = torch.randn((1, 1, 28, 28))
    # ori = count_target_ops(model, example_inputs)
    # imp = tp.importance.RandomImportance()
    # ignored_layers = []
    # for m in model.modules():
    #     if isinstance(m, torch.nn.Linear) and m.out_features == 10:
    #         ignored_layers.append(m) 
    


    # pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
    #     model,
    #     example_inputs,
    #     importance=imp,
    #     iterative_steps = 200,
    #     pruning_ratio=0.99, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    #     # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
    #     ignored_layers=ignored_layers,
    # )
    # for i in range(200):
    #     pruner.step()
    #     b = sim(model, 0.9)
    #     c =  count_target_ops(model, example_inputs)[1] / ori[1]
    # pass # 5 15 114 79 15 10

