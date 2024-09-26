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

# record the gram matrix of X and Y
K_Xs = []
K_Ys = []
def IXZ_IZY_renyi(x, y, z, alpha = 1.01):
    # z is output from pruned layer 
    renyi_est = RW.renyi_estim(5, 480, DEVICE)
    Ixz, Izy = 0, 0
    b = x.shape[0]
    z = z.view(b, z.shape[1], -1)
    sigmatracer = Renyi.sigmaTracer(layerNames=["z"])
    total_Hx, total_Hy = 0, 0

    # choosing the sigma of z
    for i in range(b):
        Renyi.N_matrix(z[i], k_y=K_Ys[i], sigmatracer=sigmatracer, 
                                        iteration= i, layer = "z" , device = DEVICE)
        renyi_est.kernel_mat(z[i], K_Xs[i], K_Ys[i], epoch = i, idx = 1)
        
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


wickSigma = -1
def wick_IXZ_IZY(x, y, z):
    global wickSigma
    batch_num = x.shape[0]
    batch_size = x.shape[1]
    IXZ, IZY = 0, 0
    renyi_wick = RW.renyi_estim(1, batch_num, DEVICE)
    for i in range(batch_num):
        _ = renyi_wick.kernel_mat(z[i].reshape(batch_size, -1), K_Xs[i], K_Ys[i], epoch=i, idx=0, factor=1)

    sigma_z = renyi_wick.getLastSigma(0)
    if wickSigma == -1:
        wickSigma = sigma_z
    rangee = range(batch_num-1, batch_num)
    for i in rangee:
        k_z = renyi_wick.kernel_mat(z[i].reshape(batch_size, -1), [], [], sigma=sigma_z, epoch=0, idx=0, factor=1)
        h_z = renyi_wick.entropy(k_z)
        h_x = renyi_wick.entropy(K_Xs[i])
        h_y = renyi_wick.entropy(K_Ys[i])

        j_XZ = renyi_wick.entropy(K_Xs[i], k_z)
        j_ZY = renyi_wick.entropy(k_z, K_Ys[i])

        IXZ = h_x + h_z - j_XZ
        IZY = h_z + h_y - j_ZY


        pass


    return IXZ, IZY





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
                

def TC_renyi_wick(Z):
    # hole the layer sigma

    sigma_z = wickSigma
    batch_num = Z.shape[0]
    batch_size = Z.shape[1]
    node_num = Z.shape[2]

    rangee = range(batch_num-1, batch_num)
    renyi_wick = RW.renyi_estim(1, 1, DEVICE)

    TC = 0
    for i in rangee:
        H_zj = torch.zeros([node_num], dtype=torch.float64).to(DEVICE)
        HM = None

        for j in range(node_num):
            k_zj = renyi_wick.kernel_mat(Z[i, :, [j]].reshape(batch_size, -1), [], [], sigma=sigma_z, epoch=0, idx=0, factor=1)

            # calculating joint Z_k
            if type(HM) == type(None):
                HM = k_zj
            else:
                HM = HM.mul(k_zj)
            
            H_zj[j] = renyi_wick.entropy(k_zj)
    
        J_zj = renyi_wick.entropy(HM)
        TC += H_zj.sum() - J_zj
    return TC / len(list(rangee))

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

if __name__=="__main__":
    from MNIST_datasets import MNIST_Dataset
    from Networks.LeNet import LeNet6_16_AllTanh, LeNet7_16_AllTanh, LeNet8_16_AllTanh, LeNet9_16_AllTanh
    from pretrained_models import Loads
    from prune_utils.prune_method import PruningMethod
    from save_stdout import SaveOutput
    
    from information.information_process_bins import extract_p, entropy_p, mutual_information
    import information.Renyi as Renyi
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type = str, default = 'MNIST10C_LeNet6_16_AllTanh_Val_conv1')
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--layer', type = str, default = "")
    parser.add_argument('--sampling', type = str, default = "reyei_uns_tot")
    parser.add_argument('--IZYIXZsampling', type = str, default = "reyei_uns_tot")
    # the method for calculating the entropy 
    # bin:              binning

    # reyei_uns:        Renyi unsupervised method (z1 and z2 allign to 
    #                   layer, sigma of layer chosen with the method 
    #                   from Information Flows of Diverse Autoencoders

    parser.add_argument('--only_acc', type = int, default = 0) 
    parser.add_argument('--batch_size', type = int, default = 256)  
    parser.add_argument('--sigma_method', type = str, default = "all",
                        help = "fix (use the method of autoencoder), remain (sample the remain Z), all (sample or Z(w/o pruned) )")
    parser.add_argument('--only_IXZIZY', type = int, default = 2) 
    parser.add_argument('--IXZ_batch', type = int, default = -1)  
    parser.add_argument("--skip_zero", type=int, default=0)
    parser.add_argument("--total_batch", type=int, default=0)
               
    parser.add_argument("--split", type=float, default=1)
    parser.add_argument("--rev", type=int, default=0)
    args = parser.parse_args()
    
    DEVICE = f'cuda:{args.device}'

     # 建立 30 個 bin (經驗法則選取)
    BINS = torch.from_numpy(np.linspace(-1, 1, 30)).to(DEVICE)
    
    # 讀取模型資料 ====================================================================
    rev_name = "_rev" if args.rev == 1 else ""
    rev_name += "_skip_zero" if args.skip_zero == 1 else ""
    rev_name += "_batchMean" if args.total_batch == 1 else ""
    if args.split == 1 and args.rev == 0:
        args.split = ""
        rev_name = ""
    classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, modelSavePath = \
    Loads(args.case, test=True, layer=args.layer, batchSize=args.batch_size, sampling = args.sampling  , split = str(args.split) + rev_name)
    
    print("\n" + str(args.IXZ_batch) + "\n")
    batch_size = args.IXZ_batch if args.IXZ_batch > 0 else batch_size

    model = model.to(DEVICE)
    sys.stdout = SaveOutput(stdout_path)
    

    checkpoint   = torch.load(modelSavePath, map_location = DEVICE)
    # checkpoint   = torch.load(m_path, map_location = DEVICE)

    train_dataset = checkpoint['train_dataset']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = checkpoint['test_dataset']
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
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
    for linear_idx, layer_name in linear_layers.items():
        # 如果有指定 layer，就去看目前執行的 layer 是不是指定的，沒有就跳過
        sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero")
        if f'filter_delete_order_{sample_name}_{layer_name}' not in checkpoint.keys():
            if f'filter_delete_order_{layer_name}' not in checkpoint.keys():
                print(">>> No pruning order")
                continue
        
        # 蒐集每個 batch 、目前指定的 layer 的輸出以及 X 與 Y
        Xs = []
        Ys = []
        Zs = []
        for idx, (train_x, train_label) in enumerate(train_loader):
            Xs.append(train_x)
            Ys.append(train_label)
            train_x = train_x.to(DEVICE)
            model.eval()
            _, Z = model(train_x.float())
            Zs.append(Z[linear_idx].detach().cpu())

        # 處理資料形狀 ==========================================================
        if args.sampling in ["bin", "SMI"]:
            X_train = torch.cat(Xs, dim=0).to(DEVICE)
            Y_train = torch.cat(Ys, dim=0).to(DEVICE)
            Z_train = torch.cat(Zs, dim=0).to(DEVICE)

            # 記錄目前 node 的數量
            node_num = Z_train.shape[1]
        
        # Renyi 的處理
        else:
            # 資料形狀，如果最後一個batch不完整（不等於batch size），就捨棄
            try:
                X_train = torch.stack(Xs).to(DEVICE)
                Y_train = torch.stack(Ys).to(DEVICE)
                Z_train = torch.stack(Zs).to(DEVICE)
            except:
                X_train = torch.stack(Xs[:-1]).to(DEVICE)
                Y_train = torch.stack(Ys[:-1]).to(DEVICE)
                Z_train = torch.stack(Zs[:-1]).to(DEVICE)

            
            if args.IZYIXZsampling in ["renyi_wick"]:
                renyi_wick = RW.renyi_estim(1, 1, DEVICE)

            # 記錄目前 node 的數量
            node_num = Z_train[-1].shape[1]
            if len(Z_train.shape) == 5:
                Z_train = Z_train.reshape(Z_train.shape[0], Z_train.shape[1], Z_train.shape[2], -1)

            # 計算K_Z
            for z in Zs:
                K_Zs.append( Renyi.gaussKernel(z.detach().to(DEVICE), sigma = _compute_sigma(z.detach()) ) )

            # 計算K_X
            for x in Xs:
                if args.IZYIXZsampling not in ["renyi_wick"]:
                    N_x = Renyi.N_matrix(x.float(), sigma=torch.tensor(8.0), device = DEVICE)
                else:
                    N_x = renyi_wick.kernel_mat(x, [], [], sigma=torch.tensor(8.0).to(DEVICE), factor=1)
                K_Xs.append(N_x)
            y_hots = []
            # 計算K_Y
            for y in Ys:
                # 轉成 one hot ========================================================================
                y_hot = torch.zeros((y.size(0), torch.max(y).int()+1)).to(DEVICE)
                for j in range(y.size(0)):
                    y_hot[j, y[j].int()] = 1
                y_hots.append(y_hot)
                if args.IZYIXZsampling not in ["renyi_wick"]:
                    N_y = Renyi.N_matrix(y_hot.float(), sigma=torch.tensor(0.1), device = DEVICE, activate=False)
                else:
                    N_y = renyi_wick.kernel_mat(y_hot, [], [], sigma=torch.tensor(0.1).cuda(), factor=1)
                K_Ys.append(N_y)


        print(f"----------- {layer_name} Filter Test. -----------")
        sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero")
        if f'filter_delete_order_{sample_name}_{layer_name}' not in checkpoint.keys():
            print(">> Try to load order which is no name")
            filter_delete_order = checkpoint[f'filter_delete_order_{layer_name}']
            print(f'>> Filter_delete_order_{layer_name}')
        else:
            filter_delete_order = checkpoint[f'filter_delete_order_{sample_name}_{layer_name}']
            print(f'>> Filter_delete_order_{sample_name}_{layer_name}')
        # print("強制使用輸入順序： [14, 3, 2, 13, 1, 8, 0, 7, 11, 15, 12, 5, 10, 6, 4, 9]")
        # filter_delete_order =  [14, 3, 2, 13, 1, 8, 0, 7, 11, 15, 12, 5, 10, 6, 4, 9]

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
                if(isinstance(layer_module, torch.nn.Conv2d) and name == layer_name):
                    if i==0:
                        print(layer_module)
                    PMethod.prune_nodes(layer_module, comb, node_num)

            # 計算TC IXZ IZY

            acc, TC, Ixz, Izy, Hx, Hy, IZkZnok = -1, -1, -1, -1, -1, -1, -1
            if args.only_acc == 0:
                if args.IZYIXZsampling == "bin":
                    if args.only_IXZIZY == 0:
                        TC = Total_Correlation_bin(Z_train[:, comb]).item()
                    else:
                        TC = -1
                    if args.only_IXZIZY != 2:
                        Ixz, Izy = IXZ_IZY_bin(X_train, Y_train, Z_train[:, comb])
                    if i != 0:
                        IZkZnok = MI_bin(Z_train[:, [filter_delete_order[i-1]] ], Z_train[:, comb])
                elif args.IZYIXZsampling == "reyei_uns":
                    if args.only_IXZIZY != 1:
                        TC = Total_Correlation_renyi(Z_train[:, :, comb], comb).item()
                    else:
                        TC = -1
                    Ixz, Izy, Hx, Hy = IXZ_IZY_renyi(X_train, Y_train, Z_train[:, :, comb])

                elif args.IZYIXZsampling in ["renyi_wick"]:
                    Ixz, Izy = wick_IXZ_IZY(X_train, Y_train, Z_train[:, :, comb])
                    TC = TC_renyi_wick(Z_train[:, :, comb])
                    Hx = -1
                    Hy = -1
                else:
                    TC = -1
                    Ixz = -1
                    Izy = -1
                    Hx = -1
                    Hy = -1 
            else:
                TC = -1
                Ixz = -1
                Izy = -1
                Hx = -1
                Hy = -1

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
                pass
            acc = all_correct_num / all_sample_num

            # 印出來
            try:
                print(f"[remain {len(comb):>2d}nodes]test acc: {acc:.4f}, TC: {TC:.4f}, {Ixz:.4f}, {Izy:.4f}, Hx {Hx:.4f}, Hy {Hy:.4f}, I(Z_k;Z_nok) {IZkZnok:.4f}")
            except:
                try:
                    print(f"[remain {len(comb):>2d}nodes]test acc: {acc:.4f}, TC: {TC:.4f}, {Ixz:.4f}, {Izy:.4f}, Hx {Hx:.4f}, Hy {Hy:.4f}")
                except:
                    print(f"[remain {len(comb):>2d}nodes]test acc: {acc:.4f}, TC: {TC:.4f}, {Ixz:.4f}, {Izy:.4f}")
            total_preds[len(comb)] = {"predicts" : predicts, "ground_truth" : ground_truth}
            torch.save(total_preds, stdout_path.replace(".txt", "_preds.ckpt"))
            pass
    