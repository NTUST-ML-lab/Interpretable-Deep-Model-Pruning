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
from sys_util import checkFile, existFile

def _compute_sigma(x, gamma = 1.0, normalize_dimension = True):
    '''
    Calculate the width of guass kernel
    x: data with shape (nums of data, features)
    Output: sigma for guass kernel width
    '''
    # Copy from offical code of paper: Information Flows of Diverse Autoencoders

    # Add support for cnn (flatten the width and height)
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
    pz1, unique_inverse_z1 = extract_p(digitized_z1)
    pz2, unique_inverse_z2 = extract_p(digitized_z2)

    # Calculate the mutual information
    Hz1, Hz1_z2, Iz1z2 = mutual_information(digitized_z1, pz1, pz2, unique_inverse_z2)
    return Iz1z2

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

def MI_reyei_ST(z1, z2, sigmatracer_tmp, alpha = 1.01):
    '''
    return I(z1; z2) using alpha-Renyi

    sigmatracer_tmp shall be contained "z1_name" and 'z2'
    '''

    # K_z: Kernel matrix of layer
    global K_z

    b = z1.shape[0] # nums of batch

    # chosing the sigma ================================================
    for i in range(b):
        z1_k = Renyi.gaussKernel(z1[i], k_y=K_z[i], sigmatracer=sigmatracer_tmp, 
                                    iteration= i, layer = "z1", device = DEVICE)
        z2_k = Renyi.gaussKernel(z2[i], k_y=K_z[i], sigmatracer=sigmatracer_tmp, 
                                    iteration= i, layer = "z2", device = DEVICE)
    # get the sigma of z1 and z2
    sigma_z1 = sigmatracer_tmp.getLast("z1")
    sigma_z2 = sigmatracer_tmp.getLast("z2")
    # ==================================================================
    
    # calculating the I(z1; z2) ========================================
    MI_sum = 0
    for idx in range(b):
        # calculating the Gram matrix ==================================
        Kz1 = Renyi.N_matrix(z1[idx], sigma=sigma_z1)
        Kz2 = Renyi.N_matrix(z2[idx], sigma=sigma_z2)

        # the joint guass kernel of z1, z2
        hm = Kz1.mul(Kz2)

        # calculating the H(z1), H(z2), H(z1,z2)
        # the operation of kernel / kernel.trace() is normalization
        Hz1 = Renyi.Entropy2(Kz1 / Kz1.trace(), alpha, device = DEVICE)
        Hz2 = Renyi.Entropy2(Kz2 / Kz2.trace(), alpha, device = DEVICE)
        JE = Renyi.Entropy2(hm / hm.trace(), alpha, device = DEVICE)

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

    global args

    # get the shape of layer output
    k = Z.shape[-1]              
    MaxMI = -float('inf')
    K = []
    print(f'[{k}]')

    # calculating the I(Z_k; Z_noK) from k = 0 ~ nums of node
    # note: here we use i to specify k 
    for i in range(k):
        temp_remain = [j for j in range(k) if j!=i]
        if args.sampling == "bin":
            Z_k = Z[:, [i]]             # 抓取 Z_K
            Z_nok = Z[:, temp_remain]   # 抓取 Z_noK
            M = MI_bin(Z_k, Z_nok)      # 計算 MI
        elif args.sampling == "renyi":
            Z_k = Z[:, :, [i]]             # 抓取 Z_K
            Z_nok = Z[:, :, temp_remain]   # 抓取 Z_noK
            M = MI_reyei(Z_k, Z_nok)
        elif args.sampling == "reyei_sampling" or args.sampling == "reyei_uns" :
            Z_k = Z[:, :, [i]]             # 抓取 Z_K
            Z_nok = Z[:, :, temp_remain]   # 抓取 Z_noK
            sigmatracer_tmp = Renyi.sigmaTracer(layerNames=["z1", "z2"])
            M = MI_reyei_ST(Z_k, Z_nok, sigmatracer_tmp)

        elif args.sampling == "p_reyei":
            Z_k = Z[:, :, [i]]             # 抓取 Z_K
            Z_nok = Z[:, :, temp_remain]   # 抓取 Z_noK
            M = MI_p_reyei(Z_k, Z_nok)
        
        elif args.sampling == "renyi_IZY":
            Z_k = Z[:, :, [i]]             # 抓取 Z_K
            Z_nok = Z[:, :, temp_remain]   # 抓取 Z_noK

            zk_name = f"z_{remain_nodes[i]}"
            if zk_name not in Renyi_IZY:
                # chosing the sigma ================================================
                sigmatracer_tmp = Renyi.sigmaTracer(layerNames=["z1"])
                b = Z_k.shape[0]
                for idx in range(b):
                    z1_k = Renyi.gaussKernel(Z_k[idx], k_y=K_y[idx], sigmatracer=sigmatracer_tmp, 
                                                    iteration= idx, layer = "z1", device = DEVICE)
                sigma_z1 = sigmatracer_tmp.getLast("z1")

                tmp_m = 0
                for idx in range(Z_k.shape[0]):
                    tmp_k = Renyi.N_matrix(Z_k[idx], sigma=sigma_z1).to(DEVICE) ################## ?????????????????????????????????????????????????
                    hm = tmp_k.mul(K_y[idx])
                    tmp_m += Renyi.Entropy2(tmp_k / tmp_k.trace(), alpha=1.01) + Renyi.Entropy2(K_y[idx] / K_y[idx].trace(), alpha=1.01) - Renyi.Entropy2(hm / hm.trace(), alpha=1.01)

                M = tmp_m / Z_k.shape[0] * -1
                Renyi_IZY[zk_name] = M
            else:
                M = Renyi_IZY[zk_name]

        if (i+1)%100==0:
            print(i+1)
        elif (i+1)%5==0:
            print(i+1, end='', flush=True)
        else:
            print('.', end='', flush=True)
        
        # Find the one with the largest MI, and collect them together if there is any MI value. 
        if M > MaxMI:
            MaxMI = M
            K = [i]
        elif M == MaxMI:
            K.append(i)
    
    # Randomly select a node (with the same and maximum MI)
    choosen_k = K[np.random.randint(len(K))]
    
    # delete the chosen node
    delete_node = remain_nodes.pop(choosen_k)
    print(flush=True)

    # return Z_noK and the idx of pruned node
    if args.sampling != "bin":
        return torch.cat([Z[:, :, :choosen_k], Z[:, :, choosen_k+1:]], dim=2), remain_nodes, delete_node # 回傳Z_noK 的 Z (模擬刪除)
    return torch.cat([Z[:, :choosen_k], Z[:, choosen_k+1:]], dim=1), remain_nodes, delete_node # 回傳Z_noK 的 Z (模擬刪除)


def Filter_get_Selections(Z, mode=None):
    '''
    Choosing the order of filters to be pruned
    '''

    remain_nodes = list(range(node_num))    # list of nodes
    delete_nodes = []                       # pruned node list
    Selections = [remain_nodes.copy()]      # record the remain_nodes by during pruning 

    sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero")
    filename = modelSavePath.split(".")[0]+"_" + args.case + "_runningTmp.ckpt"
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

args = ""
Sigma = -1
K_z = []
K_y = []
if __name__=="__main__":
    from Networks.LeNet import LeNet6_16_AllTanh, LeNet7_16_AllTanh, LeNet8_16_AllTanh, LeNet9_16_AllTanh
    from pretrained_models_CIFAR10 import Loads
    from prune_utils.prune_method import PruningMethod
    from save_stdout import SaveOutput
    
    from information.information_process_bins import extract_p, entropy_p, mutual_information
    import information.Renyi as Renyi
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type = str, default = '')
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--layer', type = str, default = "")
    parser.add_argument('--sampling', type = str, default = "bin")
    # the method for calculating the entropy 
    # bin:              binning

    # renyi:            Renyi supervised method (z1 and z2 allign to label) ***deprecated***

    # p_reyei:          Renyi with polynomial Kernel                        ***deprecated***

    # reyei_sampling:   Renyi supervised method (z1 and z2 allign 
    #                   to layer, layer allign to label)                    ***deprecated***

    # reyei_uns:        Renyi unsupervised method (z1 and z2 allign to 
    #                   layer, sigma of layer chosen with the method 
    #                   from Information Flows of Diverse Autoencoders

    # renyi_IZY:        HREL

    parser.add_argument('--batch_size', type = int, default = 100)  
    parser.add_argument("--skip_zero", type=int, default=1)
    parser.add_argument("--load", type=int, default=0)

    parser.add_argument("--split", type=float, default=1)


    args = parser.parse_args()
    
    DEVICE = f'cuda:{args.device}'

    # 建立 30 個 bin (經驗法則選取)
    BINS = torch.from_numpy(np.linspace(-1, 1, 30)).to(DEVICE)
    
    # 讀取模型資料 ====================================================================
    classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, modelSavePath = \
    Loads(args.case, layer=args.layer, sampling=args.sampling, batchSize=args.batch_size)
    
    model = model.to(DEVICE)
    sys.stdout = SaveOutput(stdout_path)
    
    checkpoint   = torch.load(m_path, map_location = DEVICE)
    
    last_epoch_imgs = checkpoint['last_epoch_imgs']
    last_epoch_labels = checkpoint['last_epoch_labels']
    
    state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key[:6] == "model.":
            state_dict[key[6:]] = value
        else:
            state_dict[key] = value
        
    model.load_state_dict(state_dict)
    backup_model = deepcopy(model)
    
    print(m_path)
    print(model)
    
    saves = checkpoint
    # ==============================================================================

    for linear_idx, layer_name in linear_layers.items():

        # 蒐集每個 batch 、目前指定的 layer 的輸出
        batchsize = batch_size
        idx = 0
        Zs = []

        if args.sampling == "renyi" or args.sampling == "reyei_sampling":
            sigmatracer = Renyi.sigmaTracer([layer_name])

        # 記錄要刪的那層的輸出 ===================================================
        iteration = 0
        
        # 模擬 batch 的選取與計算過程
        while idx < last_epoch_imgs.shape[0]-1:
            train_x = last_epoch_imgs[idx:idx+batchsize].to(DEVICE)
            train_label = last_epoch_labels[idx:idx+batchsize].to(DEVICE)
            model.eval()
            _, Z = model(train_x.float())
            Zs.append(Z[linear_idx].detach().cpu())
            idx += batchsize
            iteration += 1

            # 如果是使用 renyi 或 reyei_sampling ，會一併把 k_y k_z 給存下來 ====
            if args.sampling == "renyi" or args.sampling == "reyei_sampling":
                y_hot = torch.zeros((train_label.size(0), torch.max(train_label).int()+1)).to(DEVICE)
                for i in range(train_label.size(0)):
                    y_hot[i, train_label[i].int()] = 1
                k_y = Renyi.gaussKernel(y_hot.float(), sigma=torch.tensor(0.1), device = DEVICE, activate=True)
                k_z = Renyi.gaussKernel(Z[linear_idx].detach(), k_y=k_y, sigmatracer=sigmatracer, 
                                  iteration= iteration, layer = layer_name, device = DEVICE)
            
            elif args.sampling in ["renyi_IZY"]:
                y_hot = torch.zeros((train_label.size(0), torch.max(train_label).int()+1)).to(DEVICE)
                for i in range(train_label.size(0)):
                    y_hot[i, train_label[i].int()] = 1
                k_y = Renyi.gaussKernel(y_hot.float(), sigma=torch.tensor(0.1), device = DEVICE, activate=True)
                K_y.append(k_y)
            
            # ===================================================================
            
        # reyei_sampling 會去將 k_z 與 k_y 做對齊
        if args.sampling == "reyei_sampling":
            for z in Zs:
                K_z.append( Renyi.gaussKernel(z.detach().to(DEVICE), sigma = sigmatracer.getLast(layer_name) ) )
        
        # reyei_uns 會使用 Information Flows of Diverse Autoencoders 提及的方法計算 k_z
        elif args.sampling == "reyei_uns":
            for z in Zs:
                K_z.append( Renyi.gaussKernel(z.detach().to(DEVICE), sigma = _compute_sigma(z.detach()) ) )
        
        # 處理資料形狀 ==========================================================
        if args.sampling == "bin":
            Z_train = torch.cat(Zs, dim=0).to(DEVICE)

            # 記錄目前 node 的數量
            node_num = Z_train.shape[1]
        else:
            try:
                Z_train = torch.stack(Zs).to(DEVICE)
            except:
                Z_train = torch.stack(Zs[:-1]).to(DEVICE)
            if args.sampling == "renyi":
                Sigma = sigmatracer.getLast(layer_name)

            # 記錄目前 node 的數量
            node_num = Z_train[-1].shape[1]
        # =======================================================================

        # 開始刪 filter，使用方才模型跑出來的 layer output
        start = time.time()
        filter_delete_order = Filter_get_Selections(Z_train)
        end = time.time()
        print(f"delete_order: {filter_delete_order}")
        print(f"----------- {layer_name} Filter done. ({end-start:.1f}s)-----------")

        # 存檔，如果有路徑衝突，就會嘗試合併兩個檔案
        saves[f'filter_delete_order_{layer_name}'] = filter_delete_order
        try:
            if '.ckpt' in m_path:
                savePath = modelSavePath[:-5]+f'_filter_delete_order.ckpt'
            else:
                savePath = modelSavePath[:-5]+f'_filter_delete_order.pt'
            if existFile(savePath):
                tmp = torch.load(savePath)
                tmp[f'filter_delete_order_{layer_name}'] = filter_delete_order
                torch.save(tmp, savePath)
            else:
                torch.save(saves, savePath)
        except:
            if '.ckpt' in m_path:
                savePath = modelSavePath[:-5]+f'_filter_delete_order.ckpt'
                savePath = checkFile(savePath)
                torch.save(saves, savePath)
            else:
                savePath = modelSavePath[:-5]+f'_filter_delete_order.pt'
                savePath = checkFile(savePath)
                torch.save(saves, savePath)