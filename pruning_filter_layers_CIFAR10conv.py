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
from sys_util import checkFileName, checkFile, existFile
import gc
import information.Renyi_wick as RW



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

def MI_bin(z1, z2):
    '''
    Calculate I(z1; z2)
    '''

    # Binning the z1, z2
    digitized_z1 = BINS[torch.bucketize(z1, BINS) - 1]
    digitized_z2 = BINS[torch.bucketize(z2, BINS) - 1]

    # Calculate probability measure of z1, z2
    pz1, unique_inverse_z1 = extract_p(digitized_z1)        # samping z1 (Z_k)
    pz2, unique_inverse_z2 = extract_p(digitized_z2)        # samping z2 (Z_noK)

    # Calculate the mutual information
    Hz1, Hz1_z2, Iz1z2 = mutual_information(digitized_z1, pz1, pz2, unique_inverse_z2)
    return Iz1z2

def MI_renyi_wick(z1, z2):
    b = z1.shape[0]
    renyi_est_Z1 = RW.renyi_estim(1, b, DEVICE)
    renyi_est_Z2 = RW.renyi_estim(1, b, DEVICE)
    tmp_restim = RW.renyi_estim(1, 1, DEVICE)
    # allign Kz
    for i in range(b):
        renyi_est_Z1.kernel_mat(z1[i], k_x=K_x[i], k_y=K_z[i], epoch=i, idx=0)
        renyi_est_Z2.kernel_mat(z2[i], k_x=K_x[i], k_y=K_z[i], epoch=i, idx=0)
    pass

    # get Sigma
    sigma_z1 = renyi_est_Z1.getLastSigma(0)
    sigma_z2 = renyi_est_Z2.getLastSigma(0)

    # do only last batch
    last_batch = b-1

    k_z1 = tmp_restim.kernel_mat(z1[last_batch], [], [], sigma=sigma_z1, factor=1)
    k_z2 = tmp_restim.kernel_mat(z2[last_batch], [], [], sigma=sigma_z2, factor=1)

    h_z1 = RW.renyi_estim.entropy(k_z1)
    h_z2 = RW.renyi_estim.entropy(k_z2)
    j_z1z2 = RW.renyi_estim.entropy(k_z1, k_z2)

    return h_z1 + h_z2 - j_z1z2


def MI_reyei(z1, z2):
    '''
    Calculate I(z1; z2) with Renyi supervised method (z1 and z2 allign to label)
    Input: the shape of z1 and z2 is (batches, batch size, node output ... )
    *** Note: this method is deprecated ***
    '''

    global Sigma # The closest sigma for Z from Z kernel and Y kernel

    MI_sum = 0
    b = z1.shape[0] # nums of batches
    for idx in range(b):
        # Calcuate the guass kernel of z1 and z2
        Kz1 = Renyi.N_matrix(z1[idx], sigma=Sigma)
        Kz2 = Renyi.N_matrix(z2[idx], sigma=Sigma)
        
        # the joint guass kernel of z1, z2
        hm = Kz1.mul(Kz2)

        # calcuate the entropy
        # the operation of kernel / kernel.trace() is normalization
        Hz1 = Renyi.Entropy2(Kz1 / Kz1.trace(), alpha=1.01)
        Hz2 = Renyi.Entropy2(Kz2 / Kz2.trace(), alpha=1.01)
        JE = Renyi.Entropy2(hm / hm.trace(), alpha=1.01)

        MI_sum += Hz1 + Hz2 - JE
    return MI_sum / b  

# record the kernel and entropy of specify node 
K_Zjs = []
H_Zjs = []
node_index = []
def MI_reyei_ST(z1, z2, sigmatracer_tmp, alpha = 1.01, z1_name = ""):
    '''
    return I(z1; z2) using alpha-Renyi

    sigmatracer_tmp shall be contained "z1_name" and 'z2'
    '''

    # K_z: Kernel matrix of layer
    # K_Zjs, H_Zjs: kernel and entropy of specify node 
    global K_z, K_Zjs, H_Zjs 

    b = z1.shape[0] # nums of batch

    # initialize the K_Zjs and H_Zjs
    if len(K_Zjs) == 0:
        K_Zjs = [{} for i in range(b)]
        H_Zjs = [{} for i in range(b)]

    # chosing the sigma ================================================
    for i in range(b):
        # if Gram matrix of "z1_name" is recorded, skip the choosing sigma of "z1_name"
        if z1_name not in K_Zjs[i] or len(z1_name) == 0:
            z1_k = Renyi.gaussKernel(z1[i], k_y=K_z[i], sigmatracer=sigmatracer_tmp, 
                                        iteration= i, layer = z1_name, device = DEVICE, presion=typee)
            
        z2_k = Renyi.gaussKernel(z2[i], k_y=K_z[i], sigmatracer=sigmatracer_tmp, 
                                    iteration= i, layer = "z2", device = DEVICE, presion=typee)
    
    # get the sigma of z1 and z2
    if z1_name not in K_Zjs[i] or len(z1_name) == 0:
        sigma_z1 = sigmatracer_tmp.getLast(z1_name)
    sigma_z2 = sigmatracer_tmp.getLast("z2")

    # ==================================================================

    # calculating the I(z1; z2) ========================================
    MI_sum = 0
    for idx in range(b):

        # calculating the Gram matrix ==================================

        # if Kz1 is calculated, use the recorded one
        if z1_name not in K_Zjs[i] or len(z1_name) == 0:
            Kz1 = Renyi.N_matrix(z1[idx], sigma=sigma_z1, presion=typee)
            if  len(z1_name) != 0:
                K_Zjs[idx][z1_name] = Kz1
        else:
            Kz1 = K_Zjs[idx][z1_name]

        # calculating the Gram matrix of z2
        Kz2 = Renyi.N_matrix(z2[idx], sigma=sigma_z2, presion=typee)

        # the joint guass kernel of z1, z2
        hm = Kz1.mul(Kz2)
        # ==============================================================

        # calculating the H(z1), H(z2), H(z1,z2) =======================

        if args.sampling in ["reyei_uns", "reyei_sampling"]:

            # if Hz1 is calculated, use the recorded one
            # the operation of kernel / kernel.trace() is normalization
            if z1_name not in H_Zjs[idx] or len(z1_name) == 0: 
                Hz1 = Renyi.Entropy2(Kz1 / Kz1.trace(), alpha, device = DEVICE, presion=typee)
                if  len(z1_name) != 0:
                    H_Zjs[idx][z1_name] = Hz1
            else:
                Hz1 = H_Zjs[idx][z1_name]

            Hz2 = Renyi.Entropy2(Kz2 / Kz2.trace(), alpha, device = DEVICE, presion=typee)
            JE = Renyi.Entropy2(hm / hm.trace(), alpha, device = DEVICE, presion=typee)
        # ==============================================================



        
        MI_sum += Hz1 + Hz2 - JE
    return MI_sum / b   



def MI_p_reyei(z1, z2, alpha = 2):
    '''
    Calculate I(z1; z2) with Renyi 
    Using the polynomial Kernel
    *** Note: this method is deprecated ***
    '''
    MI_sum = 0
    b = z1.shape[0]
    for idx in range(b):
        Nz1 = Renyi.polynomialKernel(z1[idx])
        Nz2 = Renyi.polynomialKernel(z2[idx])
        hm = Nz1.mul(Nz2)
        Hz1 = Renyi.Entropy2(Nz1 / Nz1.trace(), alpha, device = DEVICE)
        Hz2 = Renyi.Entropy2(Nz2 / Nz2.trace(), alpha, device = DEVICE)
        JE = Renyi.Entropy2(hm / hm.trace(), alpha, device = DEVICE)
        MI_sum += Hz1 + Hz2 - JE
    return MI_sum / b 




Renyi_IZY = {}
def next_Z(Z, remain_nodes=[]):
    '''
    Choose which node shall be pruned
    Return output of pruned layer amd index of pruned node

    Input: pruned Z and list of remain nodes (indexs in original Z)

    If method is bin or smoothed, the shape of Z is (nums of data, nodes, ...)
    If method is Renyi based,     the shape of Z is (nums of batch, batch size, nodes, ...)
    '''

    global args, k_y
    
    # get the shape of layer output
    if len(Z.shape) == 5:
        k = Z.shape[2]
    elif len(Z.shape) > 2:
         k = Z.shape[1]
    else:
        k = Z.shape[-1]              

    flac = 1 if args.rev == 0 else -1
    MaxMI = -float('inf')        * flac
    K = []
    print(f'[{k}]')
    M_list = {}

    # calculating the I(Z_k; Z_noK) from k = 0 ~ nums of node
    # note: here we use i to specify k 
    for i in tqdm(range(k)):          
        temp_remain = [j for j in range(k) if j!=i]
        if len(Z.shape) == 5:
            Z_k = Z[:, :, [i]]             # get Z_K
            Z_nok = Z[:, :, temp_remain]   # get Z_noK
        else:
            if args.sampling in ["bin", "smoothed", "SMI"]:
                Z_k = Z[:, [i]]             # get Z_K
                Z_nok = Z[:, temp_remain]   # get Z_noK
            else:
                Z_k = Z[:, :, [i]]             # get Z_K
                Z_nok = Z[:, :, temp_remain]   # get Z_noK

        # calculating the  I(Z_k; Z_noK) =====================================

        if args.sampling == "bin":
            M = MI_bin(Z_k, Z_nok)      
        elif args.sampling == "renyi":
            M = MI_reyei(Z_k, Z_nok) * -1
        elif args.sampling in ["reyei_uns", "low_dim", "reyei_sampling"] :
            z1_name = f"z_{remain_nodes[i]}"
            sigmatracer_tmp = Renyi.sigmaTracer(layerNames=[z1_name, "z2"])
            M = MI_reyei_ST(Z_k, Z_nok, sigmatracer_tmp, z1_name = z1_name)
        elif args.sampling in ["renyi_wick"]:
            M = MI_renyi_wick(Z_k, Z_nok)
            pass

        elif args.sampling == "p_reyei":
            M = MI_p_reyei(Z_k, Z_nok)


        elif args.sampling == "renyi_IZY":
            zk_name = f"z_{remain_nodes[i]}"
            if zk_name not in Renyi_IZY:
                tmp_m = 0
                for idx in range(Z_k.shape[0]):
                    tmp_k = Renyi.N_matrix(Z_k[idx], sigma=Sigma)
                    hm = tmp_k.mul(K_y[idx])
                    tmp_m += Renyi.Entropy2(tmp_k / tmp_k.trace(), alpha=1.01) + Renyi.Entropy2(K_y[idx] / K_y[idx].trace(), alpha=1.01) - Renyi.Entropy2(hm / hm.trace(), alpha=1.01)

                M = tmp_m / Z_k.shape[0] * -1
                Renyi_IZY[zk_name] = M
            else:
                M = Renyi_IZY[zk_name]
        elif args.sampling == "renyi_Hz":  
            zk_name = f"z_{remain_nodes[i]}"

            if zk_name not in Renyi_Hz:
                tmp_m = 0
                b = Z_k.shape[0]
                # chosing the sigma ================================================
                sigmatracer_tmp = Renyi.sigmaTracer(layerNames=["z1"])
                for idx in range(b):
                    z1_k = Renyi.gaussKernel(Z_k[idx], k_y=K_z[idx], sigmatracer=sigmatracer_tmp, 
                                                    iteration= idx, layer = "z1", device = DEVICE)
                sigma_z1 = sigmatracer_tmp.getLast("z1")
                # ======
                
                for idx in range(b):
                    tmp_k = Renyi.N_matrix(Z_k[idx], sigma=sigma_z1)
                    tmp_m += Renyi.Entropy2(tmp_k / tmp_k.trace(), alpha=1.01)
                M = tmp_m / b * -1
                Renyi_Hz[zk_name] = M
            else:
                M = Renyi_Hz[zk_name]
        # =====================================================================

        if (i+1)%100==0:
            print(i+1)
        elif (i+1)%5==0:
            print(i+1, end='', flush=True)
        else:
            print('.', end='', flush=True)
        
        # Find the one with the largest MI, and collect them together if there is any MI value. 
        if M * flac > MaxMI * flac:           
            MaxMI = M
            K = [i]
        elif M == MaxMI:
            K.append(i)
        M_list[i] = M.tolist()

    # Randomly select a node (with the same and maximum MI)
    choosen_k = K[np.random.randint(len(K))]    
    



    # delete the chosen node
    delete_node = remain_nodes.pop(choosen_k)

    print(f"============= next to prune: idx: {choosen_k}, node: {delete_node}")
    print(M_list)

    print(flush=True)

    # return Z_noK and the idx of pruned node
    if args.sampling not in ["bin", "smoothed", "SMI"]:
        return torch.cat([Z[:, :, :choosen_k], Z[:, :, choosen_k+1:]], dim=2), remain_nodes, delete_node 
    return torch.cat([Z[:, :choosen_k], Z[:, choosen_k+1:]], dim=1), remain_nodes, delete_node 



def Filter_get_Selections(Z, mode=None):
    '''
    Choosing the order of filters to be pruned
    '''

    remain_nodes = list(range(node_num))    # list of nodes
    delete_nodes = []                       # pruned node list
    Selections = [remain_nodes.copy()]      # record the remain_nodes by during pruning 

    sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero") + ("_rev" if args.rev == 1 else "")
    filename = modelSavePath.split(".")[0]+"_" + args.case + "_" +  args.layer + "_runningTmp.ckpt"
    if args.load == 1:
        if existFile(filename):
            runningtmp = torch.load(filename)
            if sample_name in runningtmp:
                remain_nodes = runningtmp[sample_name]["remain_nodes"]
                delete_nodes = runningtmp[sample_name]["delete_nodes"]
                Selections   = runningtmp[sample_name]["Selections"]

                if args.sampling not in ["bin", "smoothed", "SMI"]:
                    Z = Z[:, :, remain_nodes ]
                else:
                    Z = Z[:, remain_nodes ]


    # check all zero node
    cnt = 0
    if args.skip_zero == 1:
        while cnt < len(remain_nodes):
            flag = False
            if args.sampling not in ["bin", "smoothed", "SMI"]:
                if torch.sum(Z[:, :, [cnt]] == 0).item() / Z[:, :, [cnt]].numel() > 0.99:
                    Z = torch.cat([Z[:, :, :cnt], Z[:, :, cnt+1:]], dim=2)
                    flag = True
            else:
                if torch.sum(Z[:, [cnt]] == 0).item() / Z[:, [cnt]].numel() > 0.99:
                    Z = torch.cat([Z[:, :cnt], Z[:, cnt+1:]], dim=1)
                    flag = True


            if flag:
                delete_nodes.append(remain_nodes.pop(cnt))
                Selections.append(remain_nodes.copy())
            else:
                cnt += 1

            pass

    # Prune from node-1 to 1 left.
    for i in tqdm(range(len(remain_nodes)-1)):             
        Z, remain_nodes, delete_node = next_Z(Z, remain_nodes=remain_nodes)
        Selections.append(remain_nodes.copy())
        delete_nodes.append(delete_node)

        save_tmp = {"remain_nodes" : remain_nodes, "Selections" : Selections, "delete_nodes" : delete_nodes}

        if existFile(filename):
            pre = torch.load(filename)
            pre[sample_name] = save_tmp
            torch.save(pre, filename)
        else:
            save = {sample_name : save_tmp}
            torch.save(save, filename)

    return delete_nodes+Selections[-1]

def val(model, val_loader, criterion):
    model.eval()
    correct_sample = 0
    total_sample = 0
    val_loss = 0
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output, _ = model(input)
        loss = criterion(output, target)
        val_loss += loss.item()
        predict = torch.argmax(output, dim=1)
        correct_sample += torch.sum(predict == target).item()
        total_sample += target.size(0)

    val_acc = correct_sample / total_sample
    val_loss /= len(val_loader)

    return val_acc, val_loss


def getLayer(idx):
    return {
        0: "features.0",
        1: "features.3",
        2: "features.7",
        3: "features.10",
        4: "features.14",
        5: "features.17",
        6: "features.20",
        7: "features.23",
        8: "features.27",
        9: "features.30",
        10: "features.33",
        11: "features.36",
        12: "features.40",
        13: "features.43",
        14: "features.46",
        15: "features.49",
    }[idx]

def getLayerAct(idx):
    return {
        0: "features.2",
        1: "features.5",
        2: "features.9",
        3: "features.12",
        4: "features.16",
        5: "features.19",
        6: "features.22",
        7: "features.25",
        8: "features.29",
        9: "features.32",
        10: "features.35",
        11: "features.38",
        12: "features.42",
        13: "features.45",
        14: "features.48",
        15: "features.51",
    }[idx]



Renyi_IZY = {}
Renyi_Hz = {}
args = ""
Sigma = -1
K_z = []
typee = None
saves = {}
K_y = []
K_x = []
if __name__=="__main__":
    from pretrained_models_CIFAR10 import Loads
    from save_stdout import SaveOutput
    from information.information_process_bins import extract_p, entropy_p, mutual_information
    
    
    import information.Renyi as Renyi
    
    from tqdm import tqdm
    from hook import register


    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type = str, default = 'CIFAR10_10C_VGG19_64_AllTanh')
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--layer', type = str, default = "0")
    parser.add_argument('--sampling', type = str, default = "reyei_uns")
    # the method for calculating the entropy 
    # bin:              binning

    # renyi:            Renyi supervised method (z1 and z2 allign to label) 

    # p_reyei:          Renyi with polynomial Kernel                        ***deprecated***

    # reyei_sampling:   Renyi supervised method (z1 and z2 allign 
    #                   to layer, layer allign to label)                    ***deprecated***

    # reyei_uns:        Renyi unsupervised method (z1 and z2 allign to 
    #                   layer, sigma of layer chosen with the method 
    #                   from Information Flows of Diverse Autoencoders

    # smoothed:         High-Dimensional Smoothed Entropy Estimation via 
    #                   Dimensionality Reduction                            ***deprecated***

    # low_dim:          Robust and Fast Measure of Information via Low-Rank 
    #                   Representation                                      ***deprecated***

    # SMI:              Sliced mutual information

    # renyi_IZY:        HREL

    # renyi_wick:        wick version


    parser.add_argument('--batch_size', type = int, default = 100)    
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--skip_zero", type=int, default=1)
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--rev", type=int, default=0)
    parser.add_argument("--twin_device", type=int, default=0)

    parser.add_argument("--split", type=float, default=0.01)
            
    args = parser.parse_args()
    args.layer = getLayerAct(int(args.layer))

    DEVICE = f'cuda:{args.device}'
    if  torch.cuda.device_count() == 1:
        args.twin_device = 0
    if args.precision == 32:
        typee = None
    if args.precision == 64:
        typee = torch.float64

    # 建立 30 個 bin (經驗法則選取)
    BINS = torch.from_numpy(np.linspace(-1, 1, 30)).to(DEVICE)
    if args.twin_device == 1:
        DEVICE = "cuda:1"
    # 讀取模型資料 ====================================================================
    classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, modelSavePath = \
    Loads(args.case, layer=args.layer, sampling=args.sampling, batchSize=args.batch_size, split=args.split, resnet18layer=args.layer)

    model = model.to(DEVICE)
    sys.stdout = SaveOutput(stdout_path)
    
    checkpoint   = torch.load(m_path, map_location = "cpu")


    test_dataset = checkpoint['test_dataset']
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    last_epoch_imgs = checkpoint['last_epoch_imgs']
    last_epoch_labels = checkpoint['last_epoch_labels']

    loss_fn = torch.nn.CrossEntropyLoss()


    state_dict = {}
    if "state_dict" not in checkpoint:
        checkpoint["state_dict"] = checkpoint["model"]

    for key, value in checkpoint["state_dict"].items():
        if key[:6] == "model.":
            state_dict[key[6:]] = value
        else:
            state_dict[key] = value
        
    model.load_state_dict(state_dict)
    backup_model = deepcopy(model)

    print(args.__dict__)
    print("\n")
    print(m_path)
    print(model)
    print(f"\n{args.sampling}\n")
    
    saves = {
        'state_dict': checkpoint["state_dict"],
        'last_train_aug_data': checkpoint['last_epoch_imgs'],
        'last_train_aug_label': checkpoint['last_epoch_labels'],
        'test_dataset': checkpoint['test_dataset'],
    }
    
    # ==============================================================================
    layer_name = args.layer

    # 蒐集每個 batch 、目前指定的 layer 的輸出
    Zs = []
    idx = 0

    # Hook ==============
    activateMap = []
    onOff = True
    def hook(module, input, output):
        global onOff
        if not onOff:
            return
        activateMap.append(output.detach().cpu())


    reg = register(["."+args.layer])
    reg.register_hook(model, hook, True)
    hooklist = reg.get_handles()

    # =======================

    if args.sampling == "renyi" or args.sampling == "reyei_sampling":
        sigmatracer = Renyi.sigmaTracer([layer_name])

    # 記錄要刪的那層的輸出 ===================================================
    Ys = []
    Xs = []
    idx = 0
    iteration = 0
    batchsize = args.batch_size
    while idx < last_epoch_imgs.shape[0] * args.split - 1:
        last = idx+batchsize
        if last > last_epoch_imgs.shape[0] * args.split:
            last = int(last_epoch_imgs.shape[0] * args.split)
        train_x = last_epoch_imgs[idx:last].to(DEVICE)
        train_label = last_epoch_labels[idx:last].to(DEVICE)
        model.eval()
        _, _ = model(train_x.float())
        idx += batchsize
        iteration += 1

        # 如果是使用 renyi 或 reyei_sampling ，會一併把 k_y k_z 給存下來 ====

        if args.sampling == "renyi" or args.sampling == "reyei_sampling":
            y_hot = torch.zeros((train_label.size(0), torch.max(train_label).int()+1)).to(DEVICE)
            for i in range(train_label.size(0)):
                y_hot[i, train_label[i].int()] = 1
            k_y = Renyi.gaussKernel(y_hot.float(), sigma=torch.tensor(0.1), device = DEVICE, activate=True, presion=typee)
            k_z = Renyi.gaussKernel(Zs[idx].detach(), k_y=k_y, sigmatracer=sigmatracer, 
                                iteration= idx, layer = layer_name, device = DEVICE, presion=typee)
            
        elif args.sampling in ["renyi_IZY"]:
            Ys.append(train_label)
        elif args.sampling in ["renyi_wick"]:
            Xs.append(train_x)


                    
        gc.collect()
        torch.cuda.empty_cache()
        # ===================================================================

    if args.twin_device == 1:
        DEVICE = "cuda:0"

    cal_item_num = (int(last_epoch_imgs.shape[0] * args.split) // args.batch_size) * args.batch_size
    Zs = torch.cat(activateMap)[:cal_item_num, ]
    if args.sampling not in ["bin", "smoothed", "SMI"]:
        Zs = Zs.view(-1, args.batch_size, Zs.shape[1], Zs.shape[2], Zs.shape[3])
    if args.sampling in ["renyi_IZY"]:
        Ys = torch.cat(Ys)[:cal_item_num, ].view(-1, args.batch_size)
        for idx in range(Ys.shape[0]):
            train_label =  Ys[idx]
            y_hot = torch.zeros((train_label.size(0), torch.max(train_label).int()+1)).to(DEVICE)
            for i in range(train_label.size(0)):
                y_hot[i, train_label[i].int()] = 1
            k_y = Renyi.gaussKernel(y_hot.float(), sigma=torch.tensor(0.1), device = DEVICE, activate=True, presion=typee)
            K_y.append(k_y)

    elif args.sampling == "renyi_wick":
        Xs = torch.cat(Xs)[:cal_item_num, ]
        Xs = Xs.view(-1, args.batch_size, Xs.shape[1], Xs.shape[2], Xs.shape[3])
        renyi_wick = RW.renyi_estim(1, 1)
        for idx in range(Xs.shape[0]):
            K_x.append(renyi_wick.kernel_mat(Xs[idx], [], [], sigma=torch.tensor(8.0).to(DEVICE), factor=1) )

    # =======================================================================

    # reyei_sampling 會去將 k_z 與 k_y 做對齊
    if args.sampling == "reyei_sampling":
        for z in Zs:
            K_z.append( Renyi.gaussKernel(z.detach().to(DEVICE), sigma = sigmatracer.getLast(layer_name) ) )

    # reyei_uns 與 low_dim 會使用 Information Flows of Diverse Autoencoders 提及的方法計算 k_z
    elif args.sampling in ["reyei_uns", "low_dim", "renyi_Hz"]:
        for z in Zs:
            K_z.append( Renyi.gaussKernel(z.detach().to(DEVICE), sigma = _compute_sigma(z.detach()), presion=typee ) )
    
    elif args.sampling == "renyi_wick":
        rest_tmp = RW.renyi_estim(1, 1, DEVICE)
        for z in Zs:
            K_z.append( rest_tmp.kernel_mat(z.detach().to(DEVICE), [], [], sigma = _compute_sigma(z.detach()) ).to(DEVICE) )


    # 處理資料形狀 ==========================================================
    if args.sampling in ["bin", "smoothed", "SMI"]:
        Z_train = Zs.to(DEVICE)
        if len(Z_train.shape) > 3:
            Z_train = Z_train.view(Z_train.shape[0], Z_train.shape[1], -1)
        # 記錄目前 node 的數量
        node_num = Z_train.shape[1]
    else:
        if type(Zs) == list and len(Zs.shape) != 5:
            try:
                Z_train = torch.stack(Zs).to(DEVICE)
            except:
                Z_train = torch.stack(Zs[:-1]).to(DEVICE)
        else:
            Z_train = Zs.to(DEVICE)
        if args.sampling == "renyi":
            Sigma = sigmatracer.getLast(layer_name)

        # 記錄目前 node 的數量
        node_num = Z_train[-1].shape[1]

    # =======================================================================
    model.to("meta")
    del model
    del Zs
    gc.collect()
    torch.cuda.empty_cache()


    # 開始刪 filter，使用方才模型跑出來的 layer output
    start = time.time()
    filter_delete_order = Filter_get_Selections(Z_train)
    end = time.time()
    print(f"delete_order: {filter_delete_order}")
    print(f"----------- {layer_name} Filter done. ({end-start:.1f}s)-----------")

    args.sampling += "" if args.skip_zero == 0 else "_skipZero"

    # 存檔，如果有路徑衝突，就會嘗試合併兩個檔案
    saves[f'filter_delete_order_{args.sampling}_{layer_name}'] = filter_delete_order
    try:
        if '.ckpt' in m_path:
            savePath = modelSavePath[:-5]+f'_filter_delete_order.ckpt'
        else:
            savePath = modelSavePath[:-3]+f'_filter_delete_order.pt'
        if existFile(savePath):
            tmp = torch.load(savePath, map_location="cpu")
            tmp[f'filter_delete_order_{args.sampling}_{layer_name}'] = filter_delete_order
            torch.save(tmp, savePath)
        else:
            torch.save(saves, savePath)
    except:
        if '.ckpt' in m_path:
            savePath = modelSavePath[:-5]+f'_filter_delete_order.ckpt'
            savePath = checkFile(savePath)
        else:
            savePath = modelSavePath[:-3]+f'_filter_delete_order.pt'
            savePath = checkFile(savePath)
        if existFile(savePath):
            tmp = torch.load(savePath, map_location="cpu")
            tmp[f'filter_delete_order_{args.sampling}_{layer_name}'] = filter_delete_order
            torch.save(tmp, savePath)
        else:
            torch.save(saves, savePath)