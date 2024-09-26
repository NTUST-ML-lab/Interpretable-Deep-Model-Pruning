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

# record the gram matrix of X and Y
K_Xs = []
K_Ys = []
def IXZ_IZY_renyi(x, y, z, alpha = 0.5):
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
      #  N_z_x = Renyi.N_matrix(z[i], sigma=_compute_sigma(z[i]), device = DEVICE)          # IXZ 用 fix 的方法 
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

    return Ixz / b, Izy / b

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
        HM = ""

        # initial the tensor joint Z_k
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
    if args.testIZY == 1:
        Hy = entropy_p(py)
        print(f"HZ: {Hz}, Hy: {Hy}")
    return Ixz, Izy

def Total_Correlation_bin(Z):
    # [Summation H(zj)] - H(Z)
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
args = ""
if __name__=="__main__":
    from Networks.LeNet import LeNet6_16_AllTanh, LeNet7_16_AllTanh, LeNet8_16_AllTanh, LeNet9_16_AllTanh
    from pretrained_models_CIFAR10 import Loads
    from prune_utils.prune_method import PruningMethod
    from save_stdout import SaveOutput
    
    from information.information_process_bins import extract_p, entropy_p, mutual_information
    import information.Renyi as Renyi

    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type = str, default = 'CIFAR10_10C_VGG19_64_AllTanh_fc2')
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--layer', type = str, default = "")
    parser.add_argument('--sampling', type = str, default = "bin")
    # the method for calculating the entropy 
    # bin:              binning

    # reyei_uns:        Renyi unsupervised method (z1 and z2 allign to 
    #                   layer, sigma of layer chosen with the method 
    #                   from Information Flows of Diverse Autoencoders

    parser.add_argument('--only_acc', type = int, default = 1)  
    parser.add_argument('--batch_size', type = int, default = 256)  
    parser.add_argument('--testIZY', type = int, default = 0)  
    parser.add_argument('--sigma_method', type = str, default = "all",
                        help = "fix (use the method of autoencoder), remain (sample the remain Z), all (sample or Z(w/o pruned) )")
    parser.add_argument('--only_IXZIZY', type = int, default = 0) 
    parser.add_argument('--IXZ_batch', type = int, default = -1)  

    parser.add_argument("--skip_zero", type=int, default=1)
    parser.add_argument("--load", type=int, default=0)

    parser.add_argument("--split", type=float, default=1)

    
    args = parser.parse_args()
    
    DEVICE = f'cuda:{args.device}'

    # 建立 30 個 bin (經驗法則選取)
    BINS = torch.from_numpy(np.linspace(-1, 1, 30)).to(DEVICE)
    
    # 讀取模型資料 ====================================================================
    classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, modelSavePath = \
    Loads(args.case, test=True, batchSize=args.batch_size, sampling = args.sampling , split=args.split  )
    
    batch_size = args.IXZ_batch if args.IXZ_batch > 0 else batch_size

    model = model.to(DEVICE)
    sys.stdout = SaveOutput(stdout_path)
    if args.testIZY == 0:
        checkpoint   = torch.load(modelSavePath, map_location = DEVICE)
    else:
        checkpoint   = torch.load(m_path.replace("_filter_delete_order", ""), map_location = DEVICE)
    
    last_epoch_imgs = checkpoint['last_epoch_imgs']
    last_epoch_labels = checkpoint['last_epoch_labels']
    
    test_dataset = checkpoint['test_dataset']
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key[:6] == "model.":
            state_dict[key[6:]] = value
        else:
            state_dict[key] = value
        
    model.load_state_dict(state_dict)
    backup_model = deepcopy(model)
    
    # ==============================================================================

    print(model)
    # print('HY', entropy_y(last_epoch_labels))
    
    PMethod = PruningMethod()
    for linear_idx, layer_name in linear_layers.items():
        # 如果有指定 layer，就去看目前執行的 layer 是不是指定的，沒有就跳過
        if ( f'filter_delete_order_{layer_name}' not in checkpoint.keys() ) and args.testIZY == 0:
            continue
        
        # 蒐集每個 batch 、目前指定的 layer 的輸出以及 X 與 Y
        Xs = []
        Ys = []
        Zs = []
        batchsize = batch_size
        idx = 0

        # 模擬 batch 的選取與計算過程
        while idx < last_epoch_imgs.shape[0]-1 and "conv" not in args.case:
            train_x = last_epoch_imgs[idx:idx+batchsize].to(DEVICE)
            Xs.append(train_x)
            train_label = last_epoch_labels[idx:idx+batchsize].to(DEVICE)
            Ys.append(train_label)
            model.eval()
            _, Z = model(train_x.float())
            Zs.append(Z[linear_idx].detach().cpu())
            idx += batchsize

        # 處理資料形狀 ==========================================================
        if args.sampling == "bin" or args.testIZY == 1:
            X_train = last_epoch_imgs.to(DEVICE)
            Y_train = last_epoch_labels.to(DEVICE)
            Z_train = torch.cat(Zs, dim=0).to(DEVICE)
            
            node_num = Z_train.shape[1]

            if args.testIZY == 1:
                print(f"----------- {layer_name} Ixz, Izy Test. -----------")
                Ixz, Izy = IXZ_IZY_bin(X_train, Y_train, Z_train)
                print(f"Ixz: {Ixz}, Izy: {Izy}")
                break
        
        # Renyi 的處理
        elif len(Zs) != 0:
            # 資料形狀，如果最後一個batch不完整（不等於batch size），就捨棄
            try:
                X_train = torch.stack(Xs).to(DEVICE)
                Y_train = torch.stack(Ys).to(DEVICE)
                Z_train = torch.stack(Zs).to(DEVICE)
            except:
                X_train = torch.stack(Xs[:-1]).to(DEVICE)
                Y_train = torch.stack(Ys[:-1]).to(DEVICE)
                Z_train = torch.stack(Zs[:-1]).to(DEVICE)

            # 記錄目前 node 的數量
            node_num = Z_train[-1].shape[1]

            # 計算K_Z
            for z in Zs:
                K_Zs.append( Renyi.gaussKernel(z.detach().to(DEVICE), sigma = _compute_sigma(z.detach()) ) )

            # 計算K_X
            for x in Xs:
                N_x = Renyi.N_matrix(x.float(), sigma=torch.tensor(8.0), device = DEVICE)
                K_Xs.append(N_x)

            # 計算K_Y
            for y in Ys:
                # 轉成 one hot ========================================================================
                y_hot = torch.zeros((y.size(0), torch.max(y).int()+1)).to(DEVICE)
                for j in range(y.size(0)):
                    y_hot[j, y[j].int()] = 1

                N_y = Renyi.N_matrix(y_hot.float(), sigma=torch.tensor(0.1), device = DEVICE, activate=True)
                K_Ys.append(N_y)


        print(f"----------- {layer_name} Filter Test. -----------")
        filter_delete_order = checkpoint[f'filter_delete_order_{layer_name}']
        print(f"delete_order: {filter_delete_order}")

        total_preds = {}
        # 開始刪node
        for i in range(node_num):
            comb = filter_delete_order[i:]
            model = deepcopy(backup_model).to(DEVICE)

            # 刪除選取的 node，使用的是HREL的刪法
            for name, layer_module in model.named_modules():
                if(isinstance(layer_module, torch.nn.Linear) and name == layer_name):
                    if i==0:
                        print(layer_module)
                    PMethod.prune_nodes(layer_module, comb, node_num)

            # 計算TC IXZ IZY
            if args.only_acc == 0:
                if args.sampling == "bin":
                    TC = Total_Correlation_bin(Z_train[:, comb]).item()
                    Ixz, Izy = IXZ_IZY_bin(X_train, Y_train, Z_train[:, comb])
                elif args.sampling == "reyei_uns":
                    if args.only_IXZIZY != 1:
                        TC = Total_Correlation_renyi(Z_train[:, :, comb], comb).item()
                    else:
                        TC = -1
                    Ixz, Izy = IXZ_IZY_renyi(X_train, Y_train, Z_train[:, :, comb])
            else:
                TC = -1
                Ixz = -1
                Izy = -1

            # 測試，計算accuracy    
            model.eval()
            all_correct_num = 0
            all_sample_num = 0
            predicts = []
            ground_truth = []
            for idx, (test_x, test_label) in enumerate(test_loader):
                test_x = test_x.to(DEVICE)
                test_label = test_label.to(DEVICE)
                predict_y, _ = model(test_x)
                predict_y = predict_y.detach()
                predict_y =torch.argmax(predict_y, dim=-1)
                current_correct_num = predict_y == test_label
                all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
                all_sample_num += current_correct_num.shape[0]
                predicts.extend(predict_y.cpu().tolist())
                ground_truth.extend(test_label.cpu().tolist())
            acc = all_correct_num / all_sample_num
            
            # 印出來
            print(f"[remain {len(comb):>2d}nodes]test acc: {acc:.4f}, TC: {TC:.4f}, {Ixz:.4f}, {Izy:.4f}")
            total_preds[len(comb)] = {"predicts" : predicts, "ground_truth" : ground_truth}
            torch.save(total_preds, stdout_path.replace(".txt", "_preds.ckpt"))
            pass