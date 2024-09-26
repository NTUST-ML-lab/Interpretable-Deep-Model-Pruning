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
from torchstat import stat


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
    elif "resnet" in args.case:
        sgd = torch.optim.SGD(model.parameters(), lr,
                            momentum=0.9,
                            weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(sgd,
                                                            milestones=[25, 37] )
    else:
        sgd = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = None
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    Losses = []
    for current_epoch in range(epoch_num):
        model.train()
        for idx, (train_x, train_label) in tqdm(enumerate(train_loader)):
            train_x = train_x.to(DEVICE)
            train_label = train_label.to(DEVICE)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            Losses.append(loss.item())
            loss.backward()
            sgd.step()
        Loss = sum(Losses)/len(Losses)
        acc, preds = testing(True)
        
        print(f"[epoch {current_epoch+1:>2d}] training loss: {Loss:.4f}, test acc: {acc:.4f} [lr: {get_lr(sgd)}]")
        if scheduler:
            scheduler.step()
        
        if current_epoch == epoch_num -1:
            saves = {
                'state_dict': model.state_dict(),
                'test_dataset': checkpoint['test_dataset'],
                'layer_remaining_nodeN': layer_remaining_nodeN,
                "accuracy" : acc
            }
            saves.update(preds)
            savePath = modelSavePath[:-3]+ f'_finetune_last{add_str}{message}.ckpt'
            torch.save(saves, savePath)

        if acc > best_acc:
            best_acc = acc
            saves = {
                'state_dict': model.state_dict(),
                'test_dataset': checkpoint['test_dataset'],
                'layer_remaining_nodeN': layer_remaining_nodeN,
                "accuracy" : acc
            }
            saves.update(preds)
            savePath = modelSavePath[:-3]+ f'_finetune_best{add_str}{message}.ckpt'
            torch.save(saves, savePath)
        
    return acc

def testing(retPred = False):
    model.eval()
    all_correct_num = 0
    all_sample_num = 0
    predicts = []
    ground_truth = []
    for idx, (test_x, test_label) in tqdm(enumerate(test_loader)):
        test_x = test_x.to(DEVICE)
        test_label = test_label.to(DEVICE)
        predict_y = model(test_x)
        predict_y = predict_y.detach()
        predict_y =torch.argmax(predict_y, dim=-1)
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]

        predicts.extend(predict_y.cpu().tolist())
        ground_truth.extend(test_label.cpu().tolist())
        pass
    acc = all_correct_num / all_sample_num
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


def getRemainFixRatio(tRF):
    prune_item = {}
    for layer, node in layer_initN.items():
        prune_item[f"layer{layer}.0.conv1"] = int(node * tRF) if node * tRF > 1 else 1
        prune_item[f"layer{layer}.0.conv2"] = int(node * tRF) if node * tRF > 1 else 1
   #     if layer != "1":
  #          prune_item[f"layer{layer}.0.downsample.0"] = int(node * tRF) if node * tRF > 1 else 1
        prune_item[f"layer{layer}.1.conv1"] = int(node * tRF) if node * tRF > 1 else 1
        prune_item[f"layer{layer}.1.conv2"] = int(node * tRF) if node * tRF > 1 else 1
    return prune_item




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
    from FLOPsim import calRF_node, depgraph_pruning, calRF_FC, calRF_node_L1
    import information.Renyi as Renyi
    from tqdm import tqdm
    from Datasets.data_utils import get_datasets
    import torch_pruning as tp
    from resnet18_prune import resnet18Pruner
    from pytorch_util import count_target_ops
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type = str, default = 'Mini_ImageNet_resnet18_relu')
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--layer', type = str, default = "") # layer2.0.conv1
    parser.add_argument('--sampling', type = str, default = "reyei_uns")
    # the method for calculating the entropy 
    # bin:              binning

    # renyi:            Renyi supervised method (z1 and z2 allign to label) ***deprecated***

    # p_reyei:          Renyi with polynomial Kernel                        ***deprecated***

    # reyei_sampling:   Renyi supervised method (z1 and z2 allign 
    #                   to layer, layer allign to label)                    ***deprecated***

    # reyei_uns:        Renyi unsupervised method (z1 and z2 allign to 
    #                   layer, sigma of layer chosen with the method 
    #                   from Information Flows of Diverse Autoencoders

    parser.add_argument('--batch_size', type = int, default = 100)  
    parser.add_argument("--skip_zero", type=int, default=0)
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--rev", type=int, default=0)

    parser.add_argument("--split", type=float, default=0.01)

    parser.add_argument('--targetRF', type = float, nargs='+', default = [i / 100 for i in range(70, 5, -10)])
    parser.add_argument("--fix", type=int, default=0)
    parser.add_argument("--depgraph", type=int, default=0)
    parser.add_argument("--L1", type=int, default=0)
    parser.add_argument("--L1_comb", type=int, default=1)
    parser.add_argument("--only_acc", type=int, default=1)


    args = parser.parse_args()
    if args.L1 == 1 and args.depgraph == 1:
        print(" >>>>>>>>>>>>>> L1 and depgraph cannot be applied together <<<<<<<<<")
        exit()
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
        classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, layer_initN, layer_remaining_nodeN, modelSavePath = \
        Loads(args.case, test=True, finetune=True, layer=args.layer, sampling=args.sampling, batchSize= args.batch_size, split = str(args.split) + rev_name, resnet18layer=args.layer)
    stdout_path = stdout_path.replace('_filter_delete_order', f'_finetune' + ("_depGraph" if args.depgraph == 1 else "") + ("_L1" if args.L1 == 1 else "") + ("_L1_comb" if args.L1_comb == 1 else "") )

    stdout_path = stdout_path.replace('.txt', '_2.txt')
    stdout_path = check_rep(stdout_path)

    model = model.to(DEVICE)
    sys.stdout = SaveOutput(stdout_path)
    print(args.__dict__)
    checkpoint   = torch.load(modelSavePath, map_location = DEVICE)
    # checkpoint   = torch.load(m_path, map_location = DEVICE)

#     checkpoint['filter_delete_order_fc3'] = []
#     breakpoint()
#     torch.save(checkpoint, modelSavePath[:-3]+f'_filter_delete_order.pt')
    
    # train_dataset = checkpoint['train_dataset']
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = checkpoint['val_dataset']
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_loader = get_datasets(None, False, True)
    test_dataset = checkpoint['test_dataset']
    test_loader = DataLoader(test_dataset, batch_size=512 if "0" in DEVICE else 256)
    
    print()
    print(args.case.split('_')[:2])
    # print(f'number of training data: {len(train_dataset)}')
    # print(f'number of validation data: {len(val_dataset)}')
    print(f'number of test data: {len(test_dataset)}')
    
    
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
    print("classes:", classes)
    PMethod = PruningMethod()

    print(f'========== Without pruning ==========')
    acc = 0.7916 # testing()

    print(f"test acc: {acc:.4f}")
    withoutPruningAcc = acc

    Flag = 0
    Last_accs = {}
    for tRF in args.targetRF:
        if tRF > 1:
            continue
        model = deepcopy(backup_model)
        layerNode_remain_order = {}
        rp = resnet18Pruner()
        b = count_target_ops(model, example_inputs=torch.randn(1, 3, 224, 224).to(DEVICE))
        if args.depgraph == 0 and args.L1 == 0:
            sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero")
            if args.fix == 0:
                if args.L1_comb == 0:
                    print("===== Our Method, depgraph combination =====")
                    layerNode_remain = calRF_node(model, tRF, classes=100, example_inputs=torch.randn(1, 3, 224, 224), iter_num = 200, device=DEVICE)
                if args.L1_comb == 1:  
                    print("===== Our Method, L1 combination =====")
                    tmp = deepcopy(backup_model)
                    _, layerNode_remain = calRF_node_L1(tmp, tRF, torch.randn(1, 3, 224, 224), iter_num=400, device=DEVICE)
            else:
                print("===== Our Method, This combination shall not be used =====")
                layerNode_remain = getRemainFixRatio(tRF)
            if len(args.layer) == 0:
                layerNode_remain_tot = getRemainFixRatio(1)
            else:
                layerNode_remain_tot = {args.layer : 64 * int(args.layer[5])}
            for layer, remainOut in  layerNode_remain.items():
                if layer not in layerNode_remain_tot:
                    if Flag == 0:
                        print(f'>> {layer}: Skipped')
                    continue
                if "conv2" in layer:
                    layer_act = "block" + layer[5]
                else:
                    layer_act = "." + ".".join(layer.split(".")[:-1]) + ".activation"
                if f'filter_delete_order_{sample_name}_{layer_act}' not in checkpoint.keys():
                    if f'filter_delete_order_{layer_act}' not in checkpoint.keys():
                        raise Exception("Please prune the filter first")
                    if Flag == 0:
                        print(f">> {layer}: Try to load order which is no name")
                    filter_delete_order = checkpoint[f'filter_delete_order_{layer_act}']
                    if Flag == 0:
                        print(f'>> {layer}: Filter_delete_order_{layer}')
                else:
                    filter_delete_order = checkpoint[f'filter_delete_order_{sample_name}_{layer_act}']
                    if Flag == 0:
                        print(f'>> {layer}:  Filter_delete_order_{sample_name}_{layer_act}')
                layer_node = layerNode_remain_tot[layer]
                
                remain = filter_delete_order[layer_node - remainOut : ]
                removed_comb = filter_delete_order[:layer_node - remainOut]
                layerNode_remain_order[layer] = remain
                rp.prune(model, layer, removed_comb)
        elif args.L1 == 0:
            print(f'========== DepGraph Pruning ==========')
            add_str = "_DepGraph"
            if args.L1_comb == 1:
                print(f'========== L1 combination ==========')
                tmp = deepcopy(backup_model)
                L1_model, L1_remain = calRF_node_L1(tmp, tRF, torch.randn(1, 3, 224, 224), iter_num=400, device=DEVICE)
                L1_stat = count_target_ops(L1_model, example_inputs=torch.randn(1, 3, 224, 224).to(DEVICE))
                L1RP = L1_stat[1] / b[1]
                model, layerNode_remain, _ = depgraph_pruning(model, L1RP, classes=100, example_inputs=torch.randn(1, 3, 224, 224), device = DEVICE, iter_num = 400, L1_style=True)
            else:
                if "FC" not in args.case:
                    model, layerNode_remain, _ = depgraph_pruning(model, tRF, classes=100, example_inputs=torch.randn(1, 3, 224, 224), device = DEVICE, iter_num = 200, L1_style=False)
                else:
                    model, layerNode_remain, _ = depgraph_pruning(model, tRF, classes=100, example_inputs=torch.randn(1, 3, 224, 224), device = DEVICE, FConly=True, iter_num = 200)
        else:
            print(f'========== L1 (2017) Pruning ==========')
            model, layerNode_remain = calRF_node_L1(model, tRF, torch.randn(1, 3, 224, 224), iter_num=400, device=DEVICE)
        Flag += 1
        
        

        a = count_target_ops(model, example_inputs=torch.randn(1, 3, 224, 224).to(DEVICE))
        pass

        print(f'========== {tRF} Relative FLOPs ==========')
        print(layerNode_remain)
        print(f">> RF: {a[0] / b[0] :.4f}, RP: {a[1] / b[1] :.4f}")

        # model.cpu()
        # stat(model, (3, 224, 224))
        # model.to(DEVICE)
        acc = testing()
        print(f"test acc: {acc:.4f}")
        if abs(withoutPruningAcc - acc) > 0.01 or True:
            
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
    pass
    print('finetuned acc:', Last_accs)
  #  np.save(f"prune_record/batchSize{args.batch_size}/{args.sampling}/" + args.case + "_cos_finetuned acc_AllLayer.npy", Last_accs)