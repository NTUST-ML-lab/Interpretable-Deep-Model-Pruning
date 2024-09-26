import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from copy import deepcopy
import numpy as np
import sys
import time
import argparse
import torch_pruning as tp

def _compute_sigma(x, gamma = 1.0, normalize_dimension = True):
    '''
    Calculate the width of guass kernel
    x: data with shape (nums of data, features)
    Output: sigma for guass kernel width
    '''
    # Copy from offical code of paper: Information Flows of Diverse Autoencoders

    # Add support for cnn (flatten the width and height)
    if len(x.size()) == 4: 
        x = x.view(x.size()[0], -1)
    x_dims = x.shape
    n = torch.tensor(x_dims[0]).float()
    d = torch.tensor(x_dims[1]).float()
    sigma = gamma * n ** (-1 / (4 + d))
    if normalize_dimension:
        sigma = sigma * d.sqrt()
    return sigma

# record the gram matrix of X and Y
K_Xs = []
K_Ys = []
def IXZ_IZY_renyi(x, y, z, alpha = 1.01):
    # z is output from pruned layer 

    Ixz, Izy = 0, 0
    b = x.shape[0]
    sigmatracer = Renyi.sigmaTracer(layerNames=["z"])
    total_Hx, total_Hy = 0, 0

    # choosing the sigma of z
    for i in range(b):
        Renyi.N_matrix(z[i], k_y=K_Ys[i], sigmatracer=sigmatracer, 
                                        iteration= i, layer = "z" , device = DEVICE)

        
    for i in range(b):

        # IXZ ===============================================================================
    #    N_z_x = Renyi.N_matrix(z[i], sigma=_compute_sigma(z[i]), device = DEVICE)          # IXZ 用 fix 的方法 
        N_z_x = Renyi.N_matrix(z[i], sigma=sigmatracer.getLast("z"), device = DEVICE)       # 改對Y sampling

        Hx = Renyi.Entropy2(K_Xs[i] / K_Xs[i].trace(), alpha, device = DEVICE)
        Hz_x = Renyi.Entropy2(N_z_x / N_z_x.trace(), alpha, device = DEVICE)
        
        N_xz = K_Xs[i].mul(N_z_x)
        Hxz = Renyi.Entropy2(N_xz / N_xz.trace(), alpha, device = DEVICE)
        Ixz += Hx + Hz_x - Hxz
        total_Hx += Hx

        # ===================================================================================

        # IZY ===============================================================================
        N_z_y = Renyi.N_matrix(z[i], sigma=sigmatracer.getLast("z"), device=DEVICE)        # IZY 對 Y sampling

        Hy = Renyi.Entropy2(K_Ys[i] / K_Ys[i].trace(), alpha, device = DEVICE)
        Hz_y = Renyi.Entropy2(N_z_y / N_z_y.trace(), alpha, device = DEVICE)

        
        N_zy = N_z_y.mul(K_Ys[i])
        Hzy = Renyi.Entropy2(N_zy / N_zy.trace(), alpha, device = DEVICE)
        Izy += Hz_y + Hy - Hzy
        total_Hy += Hy

    return Ixz / b, Izy / b, total_Hx / b, total_Hy / b

# record the choosen sigma and gram matrix of Z without pruned
sigma_all = {}
K_Zs = []
def sampling_sigma(Z, comb):
    '''
    return sigma by given z and combination [sigma1, sigma2, ..., sigmaN]
    '''

    device = Z.device
    if len(Z.shape) == 5:
        batch, data_num, node_num, width, height = Z.shape
    else:
        batch, data_num, node_num = Z.shape
    
    # allign Z_k to remain Z (deprecated)
    if args.sigma_method == "remain":
        sigmatracer = Renyi.sigmaTracer(layerNames=["z" +str(j) for j in range(node_num)])
        for i in range(batch):
            N_z = Renyi.N_matrix(Z[i], sigma=_compute_sigma(Z[i]), device=device)
            for j in range(node_num):
                # 對 ramain Z 做 sampling ======================================================

                if len(Z.shape) == 5:
                    Renyi.N_matrix(Z[i, :, [j], :, :], k_y=N_z, sigmatracer=sigmatracer,         # maybe there is someting worng with len(z.shape) == 5
                                            iteration= i, layer = "z" + str(j), device = device)
                else:
                    Renyi.N_matrix(Z[i, :, [j]], k_y=N_z, sigmatracer=sigmatracer, 
                                            iteration= i, layer = "z" + str(j), device = device)
        

        return [sigmatracer.getLast("z" + str(j)) for j in range(node_num)]
    
    # allign Z_k to Z without pruned
    elif args.sigma_method == "all":
        sigmatracer = Renyi.sigmaTracer(layerNames=["z" +str(comb[j]) for j in range(node_num)])
        for i in range(batch):
             for j in range(node_num):
                idx = comb[j]
                # skip if calculated
                if f"z{idx}" not in sigma_all:
                    if len(Z.shape) == 5:
                        pass
                    else:
                        # choosing sigma 
                        Renyi.N_matrix(Z[i, :, [j]], k_y=K_Zs[i], sigmatracer=sigmatracer, 
                                                iteration= i, layer = "z" + str(idx), device = device)
        
        # return the sigma
        for j in range(node_num):
            idx = comb[j]
            if f"z{idx}" not in sigma_all:
                sigma_all[f"z{idx}"] = sigmatracer.getLast("z" + str(comb[j]))
                        
        return [sigma_all[f"z{comb[j]}"] for j in range(node_num)]
    
    # using the method of Information Flows of Diverse Autoencoders
    elif args.sigma_method == "fix":
        return [_compute_sigma(z[j]) for j in range(node_num)]
                


def Total_Correlation_renyi(Z, comb, alpha=1.01):
    # [Summation H(zj)] - H(Z0, ... , Zj)
    # [batch, batch_size, node]
    device = Z.device
    if len(Z.shape) == 5:
        batch, data_num, node_num, width, height = Z.shape
    else:
        batch, data_num, node_num = Z.shape

    TC = 0

    sigmas = sampling_sigma(Z, comb)

    for i in range(batch):
        # initial the array for recording H(Z_k)
        H_zj = torch.zeros([node_num], dtype=torch.float64).to(device)
        
        # initial the tensor joint Z_k
        HM = ""
        for j in range(node_num):
            sigma = sigmas[j]
            N_zj = Renyi.N_matrix(Z[i, :,[j]], sigma=sigma, device=DEVICE)

            # calculating joint Z_k
            if type(HM) == str:
                HM = N_zj
            else:
                HM = HM.mul(N_zj)

            H_zj[j] = Renyi.Entropy2(N_zj / N_zj.trace(), alpha, device = DEVICE)
            
        H_z_joint = Renyi.Entropy2(HM / HM.trace(), alpha, device = DEVICE)
        TC += H_zj.sum().to(DEVICE) - H_z_joint.to(DEVICE)
    return TC / batch



def IXZ_IZY_bin(x, y, z):
    '''
    calculate the IXZ, IZY with bin
    '''
    
    # turn y to one-hot 
    y_hot = F.one_hot(y.to(torch.int64), num_classes = 10).type(torch.float64)

    # binning the Z
    digitized_z = BINS[torch.bucketize(z, BINS) - 1]

    # calculate probability measure of x, y, z
    px, unique_inverse_x = extract_p(x)
    py, unique_inverse_y = extract_p(y_hot)
    pz, unique_inverse_z = extract_p(digitized_z)
    
    # calculate the mutual information
    Hz, Hz_x, Ixz = mutual_information(digitized_z, pz, px, unique_inverse_x)
    Hz, Hz_y, Izy = mutual_information(digitized_z, pz, py, unique_inverse_y)
    return Ixz, Izy

HZXs, HZYs = [], []
def Total_Correlation_bin(Z):
    # [Summation H(zj)] - H(Z0, ... , Zj)
    device = Z.device
    data_num, node_num = Z.shape
    
    # binning the Z
    digitized_Z = BINS[torch.bucketize(Z, BINS) - 1]
    unique_prob_Z, unique_inverse_Z = extract_p(digitized_Z)
    H_Z = entropy_p(unique_prob_Z)
    
    H_zj = torch.zeros([node_num], dtype=torch.float64).to(device)
    for j in range(node_num):
        digitized_zj = digitized_Z[...,j]
        unique_prob_zj, unique_inverse_zj = extract_p(digitized_zj)
        H_zj[j] = entropy_p(unique_prob_zj)
        
    return H_zj.sum() - H_Z

def getBlock():
    block_idx = args.layer[5:]
    return [f"layer{block_idx}.0.conv2", f"layer{block_idx}.1.conv2"]


if __name__=="__main__":
    from MNIST_datasets import MNIST_Dataset
    from Networks.LeNet import LeNet6_16_AllTanh, LeNet7_16_AllTanh, LeNet8_16_AllTanh, LeNet9_16_AllTanh
    from pretrained_models import Loads
    from prune_utils.prune_method import PruningMethod
    from save_stdout import SaveOutput
    
    from information.information_process_bins import extract_p, entropy_p, mutual_information
    import information.Renyi as Renyi
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type = str, default = 'Mini_ImageNet_resnet18_relu')
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--layer', type = str, default = "block4")
    parser.add_argument('--sampling', type = str, default = "reyei_uns")
    # the method for calculating the entropy 
    # bin:              binning

    # reyei_uns:        Renyi unsupervised method (z1 and z2 allign to 
    #                   layer, sigma of layer chosen with the method 
    #                   from Information Flows of Diverse Autoencoders

    parser.add_argument('--only_acc', type = int, default = 1) 
    parser.add_argument('--batch_size', type = int, default = 100)  
    parser.add_argument('--sigma_method', type = str, default = "all",
                        help = "fix (use the method of autoencoder), remain (sample the remain Z), all (sample or Z(w/o pruned) )")
    parser.add_argument('--only_IXZIZY', type = int, default = 1) 
    parser.add_argument('--IXZ_batch', type = int, default = -1)  
    parser.add_argument("--skip_zero", type=int, default=0)
               
    parser.add_argument("--split", type=float, default=0.01)
    parser.add_argument("--rev", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--targetNodes", type=int, default=[], nargs='+')
    parser.add_argument("--remove", type=int, default=1)
    args = parser.parse_args()
    
    DEVICE = f'cuda:{args.device}'

     # 建立 30 個 bin (經驗法則選取)
    BINS = torch.from_numpy(np.linspace(-1, 1, 30)).to(DEVICE)
    
    # 讀取模型資料 ====================================================================

    rev_name = ("_rev" if args.rev == 1 else "") + ("_rank" if args.rank == 1 else "")
    classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, modelSavePath = \
    Loads(args.case, test=True, layer=args.layer, batchSize=args.batch_size, sampling = args.sampling, split = str(args.split) + rev_name, resnet18layer=args.layer, remove = args.remove == 1)
    
    print("\n" + str(args.IXZ_batch) + "\n")
    batch_size = args.IXZ_batch if args.IXZ_batch > 0 else batch_size

    model = model.to(DEVICE)
    sys.stdout = SaveOutput(stdout_path)
    

    checkpoint   = torch.load(modelSavePath, map_location = DEVICE)
    # checkpoint   = torch.load(m_path, map_location = DEVICE)

    # train_dataset = checkpoint['train_dataset']
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    
    test_dataset = checkpoint['test_dataset']
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    dictt = {}
    for key, value in checkpoint["state_dict"].items():
        if key[:6] == "model.":
            dictt[key[6:]] = value
        else:
            dictt[key] = value

    model.load_state_dict(dictt)
    backup_model = deepcopy(model)

    # ==============================================================================

    print(model)
    

    PMethod = PruningMethod()
    if "block" not in args.layer:
        if ".activation" in args.layer:
            layer_name = ".".join(args.layer.split(".")[:-1]) + ".activation"
            args.layer = ".".join(args.layer.split(".")[:-1]) + ".conv1"
        else:
            layer_name = ".".join(args.layer.split(".")[:-1]) + ".activation"
        args.layer = args.layer if args.layer[0] != "." else args.layer[1:]
    else:
        layer_name = args.layer

    

    print(f"----------- {layer_name} Filter Test. -----------")
    sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero") 
    if f'filter_delete_order_{sample_name}_{layer_name}' not in checkpoint.keys():
        print(">> Try to load order which is no name")
        print(checkpoint.keys())
        filter_delete_order = checkpoint[f'filter_delete_order_{layer_name}']
        print(f'>> Filter_delete_order_{layer_name}')
    else:
        filter_delete_order = checkpoint[f'filter_delete_order_{sample_name}_{layer_name}']
        print(f'>> Filter_delete_order_{sample_name}_{layer_name}')
    # print("強制使用輸入順序： [14, 3, 2, 13, 1, 8, 0, 7, 11, 15, 12, 5, 10, 6, 4, 9]")
    # filter_delete_order =  [14, 3, 2, 13, 1, 8, 0, 7, 11, 15, 12, 5, 10, 6, 4, 9]

    print(f"delete_order: {filter_delete_order}")
    node_num = len(filter_delete_order)
    print("total Node:", node_num)
    # 開始刪node
    if len(args.targetNodes) == 0:
        rangee = range(1, node_num)
    else:
        rangee = args.targetNodes
    flagg = 0
    for i in rangee:
        if flagg == 0:
            print("><: ", i)
        comb = filter_delete_order[i:]
        model = deepcopy(backup_model).to(DEVICE)

        # 刪除選取的 node，使用的是HREL的刪法
        layer_names = [args.layer] if "block" not in args.layer else getBlock()

        if args.remove == 0:
            for layer in layer_names:
                for name, layer_module in model.named_modules():
                    if(isinstance(layer_module, torch.nn.Linear) and name == layer):
                        if flagg==0:
                            print(">>>", layer_module, name)
                        PMethod.prune_nodes(layer_module, comb, node_num)
                    if(isinstance(layer_module, torch.nn.Conv2d) and name == layer):
                        if flagg==0:
                            print(">>>", layer_module, name)
                        PMethod.prune_nodes(layer_module, comb, node_num)

        elif args.remove == 1:
            # # 處理batch norm

            # for layer in batchnormLayer:
            #     for name, layer_module in model.named_modules():
            #         if name == layer:
            #             pass


            # # 處理下層conv
            removed_comb = filter_delete_order[:i]
            # block_idx = args.layer[5:]
            # batchnormLayer = [f"layer{block_idx}.0.bn2", f"layer{block_idx}.1.bn2"]
            # nextConvLayer = [f"layer{block_idx}.0.conv1", f"layer{block_idx}.1.conv1"]
            # for layer in layer_names:
            #     for name, layer_module in model.named_modules():
            #         if(isinstance(layer_module, torch.nn.Conv2d) and name == layer):
            #             tp.prune_conv_out_channels(layer_module, idx = removed)

            # for layer in batchnormLayer:
            #     for name, layer_module in model.named_modules():
            #         if( name == layer):
            #             tp.prune_batchnorm_out_channels(layer_module, idx = removed)

            # for layer in nextConvLayer:
            #     for name, layer_module in model.named_modules():
            #         if(isinstance(layer_module, torch.nn.Conv2d) and name == layer):
            #             tp.prune_conv_in_channels(layer_module, idx = removed)
            DG = tp.DependencyGraph()
            DG.build_dependency(model, example_inputs=torch.randn(1, 3, 84, 84).to(DEVICE))
            layer_names = [layer_names[0]]
            for layer in layer_names:
                for name, layer_module in model.named_modules():
                    if(isinstance(layer_module, torch.nn.Conv2d) and name == layer):
                        group = DG.get_pruning_group(layer_module, tp.prune_conv_out_channels, idxs=removed_comb )
                        if DG.check_pruning_group(group):
                            group.prune()
                        if flagg == 0:
                            print(group)
                            if args.case == "Mini_ImageNet_resnet18_relu":
                                print("\n[remain 64nodes]test acc: 0.7916, TC: -1.0000, -1.0000, -1.0000, Hx -1.0000, Hy -1.0000")
            

        flagg += 1
        TC = -1
        Ixz = -1
        Izy = -1
        Hx = -1
        Hy = -1

        # 測試，計算accuracy
        model.eval()
        all_correct_num = 0
        all_sample_num = 0
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(DEVICE)
            test_label = test_label.to(DEVICE)
            predict_y = model(test_x)
            predict_y = predict_y.detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num

        # 印出來
        print(f"[remain {len(comb):>2d}nodes]test acc: {acc:.4f}, TC: {TC:.4f}, {Ixz:.4f}, {Izy:.4f}, Hx {Hx:.4f}, Hy {Hy:.4f}")


    