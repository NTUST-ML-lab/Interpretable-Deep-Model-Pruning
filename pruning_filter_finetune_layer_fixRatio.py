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
from schduler import WarmupCosineLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from sys_util import checkFile, checkFileName

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


def IXZ_IZY_renyi(z, alpha = 1.01):
    # z is output from pruned layer 

    Ixz, Izy = 0, 0
    b = z.shape[0]
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

    return (Ixz / b).cpu().tolist(), (Izy / b).cpu().tolist(), (total_Hx / b).cpu().tolist(), (total_Hy / b).cpu().tolist()

sigma_all = {}
K_Zs = {}
def sampling_sigma(Z, comb, layer, sigma_method = "all"):
    global sigma_all, K_Zs
    '''
    return sigma by given z and combination [sigma1, sigma2, ..., sigmaN]
    '''
    if layer not in sigma_all:
        sigma_all[layer] = {}

    device = Z.device
    if len(Z.shape) == 5:
        batch, data_num, node_num, width, height = Z.shape
    else:
        batch, data_num, node_num = Z.shape
    
    # allign Z_k to Z without pruned
    if sigma_method == "all":
        sigmatracer = Renyi.sigmaTracer(layerNames=["z" +str(comb[j]) for j in range(node_num)])
        for i in tqdm(range(batch)):
             for j in range(node_num):
                idx = comb[j]
                # skip if calculated
                if f"z{idx}" not in sigma_all[layer]:
                    if len(Z.shape) == 5:
                        # choosing sigma 
                        Renyi.N_matrix(Z[i, :, [j]], k_y=K_Zs[layer][i], sigmatracer=sigmatracer, 
                                                iteration= i, layer = "z" + str(idx), device = device)

                    else:
                        # choosing sigma 
                        Renyi.N_matrix(Z[i, :, [j]], k_y=K_Zs[layer][i], sigmatracer=sigmatracer, 
                                                iteration= i, layer = "z" + str(idx), device = device)
        
        # return the sigma
        for j in range(node_num):
            idx = comb[j]
            if f"z{idx}" not in sigma_all[layer]:
                sigma_all[layer][f"z{idx}"] = sigmatracer.getLast("z" + str(comb[j]))
                        
        return [sigma_all[layer][f"z{comb[j]}"] for j in range(node_num)]

def Total_Correlation_renyi(Z, comb, layer, alpha=1.01):
    # [Summation H(zj)] - H(Z0, ... , Zj)
    # [batch, batch_size, node]
    device = Z.device
    if len(Z.shape) == 5:
        batch, data_num, node_num, width, height = Z.shape
    else:
        batch, data_num, node_num = Z.shape

    TC = 0

    sigmas = sampling_sigma(Z, comb, layer)

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



def get_lr(optimizer):
    '''
    取得目前的learning rate
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']

def training(epoch_num=10, lr=1e-1, message = ""):
    global add_str
    if "VGG" in m_path:
        print("VGG optimizer")
        sgd = torch.optim.SGD(model.parameters(), lr=lr,
                              weight_decay=1e-2, momentum=0.9, nesterov=True,)
        
        total_steps = epoch_num
        scheduler = CosineAnnealingLR(sgd, total_steps, eta_min=1e-8)
        print(f"Using scheduler CosineAnnealingLR")
    else:
        sgd = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = None
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    Losses = []
    for current_epoch in range(epoch_num):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(DEVICE)
            train_label = train_label.to(DEVICE)
            sgd.zero_grad()
            predict_y, _ = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            Losses.append(loss.item())
            loss.backward()
            sgd.step()
        Loss = sum(Losses)/len(Losses)
        acc, preds = testing(True)
        
        print(f"[epoch {current_epoch+1:>2d}] training loss: {Loss:.4f}, test acc: {acc:.4f} [lr: {get_lr(sgd)}]")
        if scheduler:
            scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            saves = {
                'state_dict': model.state_dict(),
                'train_dataset': checkpoint['train_dataset'],
                'test_dataset': checkpoint['test_dataset'],
                'layerNode_remain_order': layerNode_remain_order,
                "accuracy" : acc,
                "Preds" : preds
            }
            savePath = modelSavePath[:-3]+ f'_finetune_best{add_str}{message}.ckpt'
            torch.save(saves, savePath)
        
        if current_epoch == epoch_num -1:
            saves = {
                'state_dict': model.state_dict(),
                'train_dataset': checkpoint['train_dataset'],
                'test_dataset': checkpoint['test_dataset'],
                'layerNode_remain_order': layerNode_remain_order,
                "accuracy" : acc,
                "Preds" : preds
            }
            savePath = modelSavePath[:-3]+ f'_finetune_best{add_str}{message}.ckpt'
            torch.save(saves, savePath)

    return acc


total_preds = {}
def testing(retPred = False):
    model.eval()
    all_correct_num = 0
    all_sample_num = 0
    predicts = []
    ground_truth = []
    for idx, (test_x, test_label) in tqdm(enumerate(test_loader)):
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
    total_preds[tRF] = {"predicts" : predicts, "ground_truth" : ground_truth}
    torch.save(total_preds, stdout_path.replace(".txt", "_preds.ckpt") )
    if retPred:
        return acc, {"predicts" : predicts, "ground_truth" : ground_truth}
    return acc

K_Xs = []
K_Ys = []
def get_KX_KY(train_loader):
    Xs = []
    Ys = []
    for idx, (train_x, train_label) in enumerate(train_loader):
        Xs.append(train_x)
        Ys.append(train_label)

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






def compute_Z(model, layers, train_loader, KZ = False, KXY = False):
    global K_Xs, K_Ys
    # global node_num
    Zs = {}
    Xs = []
    Ys = []

    # 跳過：renyi 與 reyei_sampling 的 sigmatracer 建立

    # total_run = args.split * len(train_loader)
    # if type(total_run) == str:
    #     total_run =  len(train_loader)


    for linear_idx, layer_name in layers.items():
        Zs[layer_name] = []
        if KZ and layer_name not in K_Zs:
            K_Zs[layer_name] = []
    for idx, (train_x, train_label) in tqdm(enumerate(train_loader)):    
        # if idx >= total_run:
        #     break
        if KXY:
            Xs.append(train_x)
            Ys.append(train_label)

        train_x = train_x.to(DEVICE)
        model.eval()
        _, Z = model(train_x.float())
        for linear_idx, layer_name in layers.items():
            Zs[layer_name].append(Z[linear_idx].detach().cpu())
    
            # 跳過：renyi 與 reyei_sampling 的 k_y k_z 儲存與計算
            # 跳過：renyi_IZY 的 k_y 儲存與計算

    if KXY:
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

    for linear_idx, layer_name in layers.items():
        try:
            Zs[layer_name] = torch.stack(Zs[layer_name]).to(DEVICE)
        except:
            Zs[layer_name] = torch.stack(Zs[layer_name][:-1]).to(DEVICE)

    if args.sampling == "reyei_sampling":
        # 跳過：reyei_sampling 會去將 k_z 與 k_y 做對齊
        pass
    elif args.sampling in ["reyei_uns", "low_dim", "renyi_Hz"]:
        for linear_idx, layer_name in layers.items():
            if Zs[layer_name].shape == 5:
                Z_sigma = _compute_sigma(Zs[layer_name].view(-1, Zs[layer_name].shape[2], Zs[layer_name].shape[3], Zs[layer_name].shape[4]))
            else:
                Z_sigma = _compute_sigma(Zs[layer_name].view(-1, Zs[layer_name].shape[2]) )
            for z in Zs[layer_name]:
                K_Zs[layer_name].append( Renyi.gaussKernel(z.detach().to(DEVICE), sigma = Z_sigma ).to(DEVICE) )


    # Z_train = {}
    # # 處理資料形狀
    # for _, layer_name in layers.items():
    #     if args.sampling in ["bin", "smoothed", "SMI"]:
    #         # 跳過
    #         pass
    #     else:
    #         try:
    #             Z_train[layer_name] = torch.stack(Zs[layer_name]).to(DEVICE)
    #         except:
    #             Z_train[layer_name] = torch.stack(Zs[layer_name][:-1]).to(DEVICE)

    #         if args.sampling == "renyi":
    #             # 跳過：sigma 計算
    #             pass

        # 記錄目前 node 的數量
        # node_num = Z_train[-1].shape[1]
    return Zs

def print_format(RF, acc = -1, TC = -1, IXZ = -1, IZY = -1, HX = -1, HY = -1, message = ""):
    print(f"[remain {RF} nodes] test acc: {acc:.4f}; {message} TC: {TC:.4f}, {IXZ:.4f}, {IZY:.4f}, Hx {HX:.4f}, Hy {HY:.4f}")


add_str = ""
if __name__=="__main__":
    try:
        from Datasets.MNIST_datasets import MNIST_Dataset
    except:
        from MNIST_datasets import MNIST_Dataset
    from Networks.LeNet import LeNet6_16_AllTanh, LeNet7_16_AllTanh, LeNet8_16_AllTanh, LeNet9_16_AllTanh
    from pretrained_models import Loads, check_rep
    import pretrained_models_CIFAR10 
    from prune_utils.prune_method import PruningMethod, pruning
    from save_stdout import SaveOutput
    from FLOPsim import calRF_node, depgraph_pruning, calRF_FC, lenetPruneSim
    import information.Renyi as Renyi
    from tqdm import tqdm
    from pytorch_util import count_target_ops
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type = str, default = 'MNIST10C_LeNet6_16_AllTanh_Val')
    parser.add_argument('--device', type = int, default = 1)
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

    parser.add_argument('--batch_size', type = int, default = 256)  
    parser.add_argument("--skip_zero", type=int, default=0)
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--rev", type=int, default=0)

    parser.add_argument("--split", type=float, default=1)

    parser.add_argument('--targetRF', type = float, nargs='+', default = [0.030749]) #[i / 100 for i in range(100, 9, -10)] + [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] )
    
    parser.add_argument("--depgraph", type=int, default=1)
    parser.add_argument("--only_acc", type=int, default=0)
    parser.add_argument("--finetune", type=int, default=1)
    parser.add_argument("--only_bigger", type=int, default=1)
    parser.add_argument("--realRF", type=int, default=0)

    args = parser.parse_args()
    args.sampling = "mixed"
    DEVICE = f'cuda:{args.device}'

    # 建立 30 個 bin (經驗法則選取)
    BINS = torch.from_numpy(np.linspace(-1, 1, 30)).to(DEVICE)
    
    # 讀取模型資料 ====================================================================
    if "VGG" in args.case:
        classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, layer_initN, layer_remaining_nodeN, modelSavePath = \
        pretrained_models_CIFAR10.Loads(args.case, test=True, finetune=True, sampling=args.sampling, batchSize = args.batch_size)
    else:
        rev_name = "_rev" if args.rev == 1 else ""
        if args.split == 1 and args.rev == 0:
            args.split = ""
            rev_name = ""
        classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, layer_initN, layer_remaining_nodeN, modelSavePath_bin = \
        Loads(args.case, test=True, finetune=True, layer=args.layer, sampling="bin")
        classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, layer_initN, layer_remaining_nodeN, modelSavePath_reyei_uns = \
        Loads(args.case, test=True, finetune=True, layer=args.layer, sampling="reyei_uns", batchSize= 100, split = str(args.split) + rev_name, resnet18layer=args.layer)
    
    if args.finetune == 1:
        stdout_path = stdout_path.replace('_filter_delete_order', f'_finetune' + ("_depGraph" if args.depgraph == 1 else "") )
    else:
        stdout_path = stdout_path.replace('_filter_delete_order', f'_without_finetune' )
    stdout_path = stdout_path.replace('.txt', '_2.txt')
    stdout_path = stdout_path.replace("batchSize100/reyei_uns", "mixed")
    stdout_path = check_rep(stdout_path)

    model = model.to(DEVICE)
    sys.stdout = SaveOutput(stdout_path)
    print(args.__dict__)
    checkpoint_bin   = torch.load(modelSavePath_bin, map_location = DEVICE)
    checkpoint_reyei_uns = torch.load(modelSavePath_reyei_uns, map_location = DEVICE)
    modelSavePath = modelSavePath_bin.replace("batchSize100/reyei_uns", "mixed")
    # checkpoint   = torch.load(m_path, map_location = DEVICE)

#     checkpoint['filter_delete_order_fc3'] = []
#     breakpoint()
#     torch.save(checkpoint, modelSavePath[:-3]+f'_filter_delete_order.pt')
    
    train_dataset = checkpoint_bin['train_dataset']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = checkpoint['val_dataset']
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = checkpoint_bin['test_dataset']
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print()
    print(args.case.split('_')[:2])
    print(f'number of training data: {len(train_dataset)}')
    # print(f'number of validation data: {len(val_dataset)}')
    print(f'number of test data: {len(test_dataset)}')
    
    
    state_dict = {}
    for key, value in checkpoint_bin["state_dict"].items():
        if key[:6] == "model.":
            state_dict[key[6:]] = value
        else:
            state_dict[key] = value
        
    model.load_state_dict(state_dict)
    backup_model = deepcopy(model)
    
    # ==============================================================================


    print(model)
    print("classes:", classes)
    PMethod = PruningMethod()

    print(f'========== Without pruning ==========')
    tRF = 1
    acc = testing()
    # get_KX_KY(train_loader)

#     Z_train = compute_Z(model, linear_layers, train_loader, KZ=True, KXY = True)
#     ip = {}
#     for linear_idx, layer_name in tqdm(linear_layers.items()): #################################### 取IXZ IZY 的時候 Z跟X跟Y要同時產出來，不能分兩次
#         comb = [i for i in range(layer_initN[layer_name])]
#         Ixz, Izy, Hx, Hy = IXZ_IZY_renyi(Z_train[layer_name][:, :, comb]) #  Z_train[:, :, comb]
#  #       TC = Total_Correlation_renyi(Z_train[layer_name], comb = comb, layer=layer_name)
#         TC = -1
#         if layer_name not in ip:
#             ip[layer_name] = {}
#         ip[layer_name]["100"] = (Ixz, Izy, Hx, Hy)
#         print_format(100, acc, TC, Ixz, Izy, Hx, Hy, message=layer_name)
#         pass
#     pass

    print(f"test acc: {acc:.4f}")
    withoutPruningAcc = acc

    Flag = 0
    Last_accs = {}
    rp_rfDICT = {}
    for tRF in args.targetRF:
        if tRF > 1:
            continue
        model = deepcopy(backup_model)
        layerNode_remain_order = {}
        b = count_target_ops(model, example_inputs=torch.randn(1, 1, 28, 28).to(DEVICE))
        if args.depgraph == 0 and tRF < 1:
            sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero")
            if "FC" not in args.case:
                layerNode_remain = calRF_node(model, tRF, classes=10 if "10C" in args.case else 4, rf= args.realRF == 1)
            else:
                layerNode_remain = calRF_FC(model, tRF, classes=10 if "10C" in args.case else 4)
            for layer, remainOut in  layerNode_remain.items():
                if layer not in layer_initN:
                    if Flag == 0:
                        print(f'>> {layer}: Skipped')
                    continue
                if "conv" in layer:
                    sample_name = "reyei_uns"
                    checkpoint = checkpoint_reyei_uns
                elif "fc" in layer:
                    sample_name = "bin"
                    checkpoint = checkpoint_bin
                if f'filter_delete_order_{sample_name}_{layer}' not in checkpoint.keys():
                    if f'filter_delete_order_{layer}' not in checkpoint.keys():
                        raise Exception("Please prune the filter first")
                    if Flag == 0:
                        print(f">> {layer}: Try to load order which is no name")
                    filter_delete_order = checkpoint[f'filter_delete_order_{layer}']
                    if Flag == 0:
                        print(f'>> {layer}: Filter_delete_order_{layer}')
                else:
                    filter_delete_order = checkpoint[f'filter_delete_order_{sample_name}_{layer}']
                    if Flag == 0:
                        print(f'>> {layer}:  Filter_delete_order_{sample_name}_{layer}')
                layer_node = layer_initN[layer]
                remain = filter_delete_order[layer_node - remainOut : ]
                layerNode_remain_order[layer] = remain
                pruning(model, layer, layer_node, remain, ifPrint=Flag == 0)
        elif  tRF < 1:
            print(f'========== DepGraph Pruning ==========')
            if "FC" not in args.case:
                model, layerNode_remain, closerest_RF = depgraph_pruning(model, tRF, classes=10 if "10" in args.case else 4, device = DEVICE, bigger=args.only_bigger == 1, iter_num=4000)
            else:
                model, layerNode_remain, closerest_RF = depgraph_pruning(model, tRF, classes=10 if "10" in args.case else 4, device = DEVICE, FConly=True)
            print(f"closerest_RF: {closerest_RF}")
            checkpoint = checkpoint_bin
        else:
            pass
        Flag += 1 if tRF != 1 else 0
        

        print(f'========== {tRF} Relative FLOPs ==========')
        if tRF < 1:
            print(layerNode_remain)
        else:
            continue
        a = count_target_ops(lenetPruneSim(len(classes), layerNode_remain['conv1']
                                           , layerNode_remain['conv2'], layerNode_remain['fc1'],
                                           layerNode_remain['fc2'], layerNode_remain['fc3']).to(DEVICE), 
                             example_inputs=torch.randn(1, 1, 28, 28).to(DEVICE))
        rp_rfDICT[tRF] = {"RF": a[0] / b[0], "RP": a[1] / b[1]}
        print(rp_rfDICT)

        acc = testing()
        print(f"test acc: {acc:.4f}")
        if abs(withoutPruningAcc - acc) > 0.01 and args.finetune == 1:
            
            # finetune
            if "VGG" in m_path:
                epoch_num, learning_rate = 50, 1e-2
            else:
                epoch_num, learning_rate = 50, 1e-1
            print("learning_rate", learning_rate)
            last_acc = training(epoch_num=epoch_num, lr=learning_rate, message=f"{tRF}")
            Last_accs[tRF] = (round(last_acc, 4))
        else:

            Last_acc = round(acc, 4)
            # Last_accs.append(f"(w/o_finetune) {Last_acc}")
            Last_accs[tRF] = (Last_acc)
    print('finetuned acc:', Last_accs)
    np.save(f"prune_record/batchSize{args.batch_size}/{args.sampling}/" + args.case + "_cos_finetuned acc_AllLayer.npy", Last_accs)