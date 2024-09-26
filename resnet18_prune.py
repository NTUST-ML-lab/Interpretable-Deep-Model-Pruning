import torch_pruning as tp
import math
try:
    import L1_prune.prune as L1
except:
    pass
class resnet18Pruner():
    def __init__(self) -> None:
        self.prune_dict = {
            "conv1" : 0
            ,"layer1.0.conv1" : 0
            ,"layer1.0.conv2" : 0
            ,"layer1.1.conv1" : 0
            ,"layer1.1.conv2" : 0

            ,"layer2.0.conv1" : 0
            ,"layer2.0.downsample.0" : 0
            ,"layer2.0.conv2" : 0
            ,"layer2.1.conv1" : 0
            ,"layer2.1.conv2" : 0

            ,"layer3.0.conv1" : 0
            ,"layer3.0.downsample.0" : 0
            ,"layer3.0.conv2" : 0
            ,"layer3.1.conv1" : 0
            ,"layer3.1.conv2" : 0

            ,"layer4.0.conv1" : 0
            ,"layer4.0.downsample.0" : 0
            ,"layer4.0.conv2" : 0
            ,"layer4.1.conv1" : 0
            ,"layer4.1.conv2" : 0
        }
    def prune(self, model, layerName, removed_comb = []):
        if self.prune_dict[layerName] != 0:
            return
        layer_map = {
            "layer1.0.conv1":         (1, "conv1", 0),
            "layer1.0.conv2":         (1, "conv2", 0),
            "layer1.1.conv1":         (1, "conv1", 1),
            "layer1.1.conv2":         (1, "conv2", 1),

            "layer2.0.conv1":         (2, "conv1", 0),
            "layer2.0.downsample.0":  (2, "conv1", 0),
            "layer2.0.conv2":         (2, "conv2", 0),
            "layer2.1.conv1":         (2, "conv1", 1),
            "layer2.1.conv2":         (2, "conv2", 1),

            "layer3.0.conv1":         (3, "conv1", 0),
            "layer3.0.downsample.0":  (3, "conv1", 0),
            "layer3.0.conv2":         (3, "conv2", 0),
            "layer3.1.conv1":         (3, "conv1", 1),
            "layer3.1.conv2":         (3, "conv2", 1),

            "layer4.0.conv1":         (4, "conv1", 0),
            "layer4.0.downsample.0":  (4, "conv1", 0),
            "layer4.0.conv2":         (4, "conv2", 0),
            "layer4.1.conv1":         (4, "conv1", 1),
            "layer4.1.conv2":         (4, "conv2", 1)
        }


        args = layer_map[layerName]
        prune_resnet(model, *args, removed_comb, self)


def prune_resnet_L1(model, tRP, device="cuda:0"):
    ori_node = [64, 64, 128, 128, 256, 256, 512, 512]
    ori_skip = [128, 256, 512]
    remainN = [ int(math.ceil(n * (1-tRP))) for n in ori_node]
    remainS = [ int(math.ceil(n * (1-tRP))) for n in ori_skip]
    pm = L1.prune_resnet(model, True, ["downsample_1", "downsample_2", "downsample_3"], remainS, True, device=device)
    pm = L1.prune_resnet(pm, True, ["conv_2", "conv_4", "conv_6", "conv_8", "conv_10", "conv_12", "conv_14", "conv_16"], 
                                remainN, False, device=device)
    return pm



def prune_resnet(model, resnetLayer, layer, blockIdx = 0, removed_comb = [], resnet18_pruner:resnet18Pruner = None):
    if resnetLayer == 1:
        {"conv1" : prune_conv1_layer1,
         "conv2" : prune_conv2_layer1}[layer](model, blockIdx, removed_comb)
    elif resnetLayer == 2:
        {"conv1" : prune_conv1_layer2,
         "conv2" : prune_conv2_layer2}[layer](model, blockIdx, removed_comb)
    elif resnetLayer == 3:
        {"conv1" : prune_conv1_layer3,
         "conv2" : prune_conv2_layer3}[layer](model, blockIdx, removed_comb)
    elif resnetLayer == 4: 
        {"conv1" : prune_conv1_layer4,
         "conv2" : prune_conv2_layer4}[layer](model, blockIdx, removed_comb)

    if resnet18_pruner is not None:
        resnet18_pruner.prune_dict[f"layer{resnetLayer}.{blockIdx}.{layer}"] = 1
        if layer == "conv2":
            resnet18_pruner.prune_dict[f"layer{resnetLayer}.0.{layer}"] = 1
            resnet18_pruner.prune_dict[f"layer{resnetLayer}.1.{layer}"] = 1
            if resnetLayer > 1:
                resnet18_pruner.prune_dict[f"layer{resnetLayer}.0.downsample.0"] = 1

def prune_conv1_layer1(model, blockIdx, removed_comb):
    tp.prune_conv_out_channels( model.layer1[blockIdx].conv1, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer1[blockIdx].bn1, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer1[blockIdx].conv2, idxs=removed_comb )

def prune_conv1_layer2(model, blockIdx, removed_comb):
    tp.prune_conv_out_channels( model.layer2[blockIdx].conv1, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer2[blockIdx].bn1, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer2[blockIdx].conv2, idxs=removed_comb )

def prune_conv1_layer3(model, blockIdx, removed_comb):
    tp.prune_conv_out_channels( model.layer3[blockIdx].conv1, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer3[blockIdx].bn1, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer3[blockIdx].conv2, idxs=removed_comb )

def prune_conv1_layer4(model, blockIdx, removed_comb):
    tp.prune_conv_out_channels( model.layer4[blockIdx].conv1, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer4[blockIdx].bn1, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer4[blockIdx].conv2, idxs=removed_comb )

def prune_conv2_layer1(model, blockIdx, removed_comb):
    tp.prune_conv_out_channels( model.conv1, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.bn1, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer1[0].conv1, idxs=removed_comb )

    tp.prune_conv_out_channels( model.layer1[0].conv2, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer1[0].bn2, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer1[1].conv1, idxs=removed_comb )

    tp.prune_conv_out_channels( model.layer1[1].conv2, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer1[1].bn2, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer2[0].conv1, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer2[0].downsample[0], idxs=removed_comb )

def prune_conv2_layer2(model, blockIdx, removed_comb):
    tp.prune_conv_out_channels( model.layer2[0].conv2, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer2[0].bn2, idxs=removed_comb )
    tp.prune_conv_out_channels( model.layer2[0].downsample[0], idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer2[0].downsample[1], idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer2[1].conv1, idxs=removed_comb )


    tp.prune_conv_out_channels( model.layer2[1].conv2, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer2[1].bn2, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer3[0].conv1, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer3[0].downsample[0], idxs=removed_comb )

def prune_conv2_layer3(model, blockIdx, removed_comb):
    tp.prune_conv_out_channels( model.layer3[0].conv2, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer3[0].bn2, idxs=removed_comb )
    tp.prune_conv_out_channels( model.layer3[0].downsample[0], idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer3[0].downsample[1], idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer3[1].conv1, idxs=removed_comb )


    tp.prune_conv_out_channels( model.layer3[1].conv2, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer3[1].bn2, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer4[0].conv1, idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer4[0].downsample[0], idxs=removed_comb )

def prune_conv2_layer4(model, blockIdx, removed_comb):
    tp.prune_conv_out_channels( model.layer4[0].conv2, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer4[0].bn2, idxs=removed_comb )
    tp.prune_conv_out_channels( model.layer4[0].downsample[0], idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer4[0].downsample[1], idxs=removed_comb )
    tp.prune_conv_in_channels( model.layer4[1].conv1, idxs=removed_comb )

    tp.prune_conv_out_channels( model.layer4[1].conv2, idxs=removed_comb )
    tp.prune_batchnorm_out_channels( model.layer4[1].bn2, idxs=removed_comb )
    tp.prune_linear_in_channels( model.fc, idxs=removed_comb )






