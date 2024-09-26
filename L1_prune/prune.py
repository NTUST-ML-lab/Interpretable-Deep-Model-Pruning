# clone from: https://github.com/EstherBear/implementation-of-pruning-filters/tree/master
# modify by Astria

import torch
import torch.nn as nn


def prune_net(net, independentflag, prune_layers, prune_channels, net_name, shortcutflag):
    print("pruning:")
    if net_name == 'vgg16':
        return prune_vgg(net, independentflag, prune_layers, prune_channels)
    elif net_name == "resnet34":
        return prune_resnet(net, independentflag, prune_layers, prune_channels, shortcutflag)
    else:
        print("The net is not provided.")
        exit(0)


def prune_vgg(net, independentflag, prune_layers, prune_channels):

    last_prune_flag = 0
    arg_index = 0
    conv_index = 1
    residue = None

    for i in range(len(net.module.features)):
        if isinstance(net.module.features[i], nn.Conv2d):
            # prune next layer's filter in dim=1
            if last_prune_flag:
                net.module.features[i], residue = get_new_conv(net.module.features[i], remove_channels, 1)
                last_prune_flag = 0
            # prune this layer's filter in dim=0
            if "conv_%d" % conv_index in prune_layers:
                remove_channels = channels_index(net.module.features[i].weight.data, prune_channels[arg_index], residue,
                                                 independentflag)
                print(prune_layers[arg_index], remove_channels)
                net.module.features[i] = get_new_conv(net.module.features[i], remove_channels, 0)
                last_prune_flag = 1
                arg_index += 1
            else:
                residue = None
            conv_index += 1
        elif isinstance(net.module.features[i], nn.BatchNorm2d) and last_prune_flag:
            # prune bn
            net.module.features[i] = get_new_norm(net.module.features[i], remove_channels)

    # prune linear
    if "conv_13" in prune_layers:
        net.module.classifier[0] = get_new_linear(net.module.classifier[0], remove_channels)
    net = net.cuda()
    print(net)
    return net


def prune_resnet(net, independentflag, prune_layers, prune_channels, shortcutflag, IFprint = False, device = "cuda:0"):
    # init
    last_prune_flag = 0
    arg_index = 0
    residue = None
   # layers = [net.module.layer1, net.module.layer2, net.module.layer3, net.module.layer4]
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]

    # prune shortcut
    if shortcutflag:
        downsample_index = 1
        for layer_index in range(len(layers)):
            for block_index in range(len(layers[layer_index])):
                if last_prune_flag:
                    # prune next block's filter in dim=1
                    layers[layer_index][block_index].conv1, residue = get_new_conv(
                        layers[layer_index][block_index].conv1, remove_channels, 1, device=device)

                if layer_index >= 1 and block_index == 0:
                    if last_prune_flag:
                        # prune next downsample's filter in dim=1
                        layers[layer_index][block_index].downsample[0], residue = get_new_conv(
                            layers[layer_index][block_index].downsample[0], remove_channels, 1, device=device)
                    else:
                        residue = None
                    if "downsample_%d" % downsample_index in prune_layers:
                        # identify channels to remove
                        remove_channels = channels_index(layers[layer_index][block_index].downsample[0].weight.data,
                                                         prune_channels[arg_index], residue, independentflag)
                        if IFprint:
                            print(prune_layers[arg_index], remove_channels)
                        # prune downsample's filter in dim=0
                        layers[layer_index][block_index].downsample[0] = get_new_conv(layers[layer_index][block_index].
                                                                                      downsample[0], remove_channels, 0, device=device)
                        # prune downsample's bn
                        layers[layer_index][block_index].downsample[1] = get_new_norm(layers[layer_index][block_index].
                                                                                      downsample[1], remove_channels, device=device)
                        arg_index += 1
                        last_prune_flag = 1
                    else:
                        last_prune_flag = 0
                    downsample_index += 1

                if last_prune_flag:
                    # prune next block's filter in dim=0
                    layers[layer_index][block_index].conv2 = get_new_conv(layers[layer_index][block_index].conv2,
                                                                          remove_channels, 0, device=device)
                    # prune next block's bn
                    layers[layer_index][block_index].bn2 = get_new_norm(layers[layer_index][block_index].bn2,
                                                                        remove_channels, device=device)
    # prune linear
    if "downsample_3" in prune_layers:
      #  net.module.fc = get_new_linear(net.module.fc, remove_channels)
      net.fc = get_new_linear(net.fc, remove_channels, device=device)

    # prune non-shortcut
    else:
        conv_index = 2
        for layer_index in range(len(layers)):
            for block_index in range(len(layers[layer_index])):
                if "conv_%d" % conv_index in prune_layers:
                    # identify channels to remove
                    remove_channels = channels_index(layers[layer_index][block_index].conv1.weight.data,
                                                     prune_channels[arg_index], residue, independentflag)
                    if IFprint:
                        print(prune_layers[arg_index], remove_channels)
                    # prune this layer's filter in dim=0
                    layers[layer_index][block_index].conv1 = get_new_conv(layers[layer_index][block_index].conv1,
                                                                          remove_channels, 0, device=device)
                    # prune next layer's filter in dim=1
                    layers[layer_index][block_index].conv2, residue = get_new_conv(
                        layers[layer_index][block_index].conv2, remove_channels, 1, device=device)
                    residue = None
                    # prune bn
                    layers[layer_index][block_index].bn1 = get_new_norm(layers[layer_index][block_index].bn1,
                                                                        remove_channels, device=device)
                    arg_index += 1
                conv_index += 2
    net = net.to(device)
    if IFprint:
        print(net)
    return net


def channels_index(weight_matrix, prune_num, residue, independentflag):
    abs_sum = torch.sum(torch.abs(weight_matrix.view(weight_matrix.size(0), -1)), dim=1)
    if independentflag and residue is not None:
        abs_sum = abs_sum + torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    _, indices = torch.sort(abs_sum)
    return indices[:prune_num].tolist()


def select_channels(weight_matrix, remove_channels, dim, device = "cuda:0"):
    indices = torch.tensor(list(set(range(weight_matrix.shape[dim])) - set(remove_channels)))
    new = torch.index_select(weight_matrix, dim, indices.to(device))
    if dim == 1:
        residue = torch.index_select(weight_matrix, dim, torch.tensor(remove_channels).to(device))
        return new, residue
    return new


def get_new_conv(old_conv, remove_channels, dim, device = "cuda:0"):
    if dim == 0:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels,
                             out_channels=old_conv.out_channels - len(remove_channels),
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data = select_channels(old_conv.weight.data, remove_channels, dim, device=device)
        if old_conv.bias is not None:
            new_conv.bias.data = select_channels(old_conv.bias.data, remove_channels, dim, device=device)
        return new_conv
    else:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels - len(remove_channels), out_channels=old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data, residue = select_channels(old_conv.weight.data, remove_channels, dim, device=device)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data
        return new_conv, residue


def get_new_norm(old_norm, remove_channels, device="cuda:0"):
    new = torch.nn.BatchNorm2d(num_features=old_norm.num_features - len(remove_channels), eps=old_norm.eps,
                               momentum=old_norm.momentum, affine=old_norm.affine,
                               track_running_stats=old_norm.track_running_stats)
    new.weight.data = select_channels(old_norm.weight.data, remove_channels, 0, device=device)
    new.bias.data = select_channels(old_norm.bias.data, remove_channels, 0, device=device)

    if old_norm.track_running_stats:
        new.running_mean.data = select_channels(old_norm.running_mean.data, remove_channels, 0, device=device)
        new.running_var.data = select_channels(old_norm.running_var.data, remove_channels, 0, device=device)

    return new


def get_new_linear(old_linear, remove_channels, device="cuda:0"):
    new = torch.nn.Linear(in_features=old_linear.in_features - len(remove_channels),
                          out_features=old_linear.out_features, bias=old_linear.bias is not None)
    new.weight.data, residue = select_channels(old_linear.weight.data, remove_channels, 1, device=device)
    if old_linear.bias is not None:
        new.bias.data = old_linear.bias.data
    return new



if __name__ == "__main__":
    from Networks.model_center import get_model
    from thop import profile
    from torchstat import stat
    class tmp():
        def __init__(self) -> None:
            self.arch = "resnet18"
            self.activate = "tanh"
            self.pretrained = False
            self.datasets = "mini_imageNet"
    
    def getRange(rangInt):
        return list(range(i for i in range(rangInt)))

    model = get_model(tmp())

    stat( model, (3, 224, 224))
    remainFP = 0.7
    ori_node = [64, 64, 128, 128, 256, 256, 512, 512]
    ori_skip = [128, 256, 512]
    remainN = [ int(n * remainFP) for n in ori_node]
    remainS = [ int(n * remainFP) for n in ori_skip]

    model.cuda()
    #                                  2.0.skip       3.0.skip         4.0.skip   
    pm = prune_resnet(model, True, ["downsample_1", "downsample_2", "downsample_3"], remainS, True)
    #                             1.0.1      1.1.1     2.0.1    2.1.1      3.0.1      3.1.1      4.0.1      4.1.1
    pm = prune_resnet(pm, True, ["conv_2", "conv_4", "conv_6", "conv_8", "conv_10", "conv_12", "conv_14", "conv_16"], 
                                remainN, False)
    model.cpu()
    stat( model, (3, 224, 224))
    
    pass