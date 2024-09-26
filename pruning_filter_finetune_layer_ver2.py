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




def get_lr(optimizer):
    '''
    取得目前的learning rate
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']

def training(epoch_num=10, lr=1e-1):
    global add_str
    if "VGG" in m_path:
        print("VGG optimizer")
        sgd = torch.optim.SGD(model.parameters(), lr=lr,
                              weight_decay=1e-2, momentum=0.9, nesterov=True,)
        
        total_steps = epoch_num
#         warmup_ratio=0.1
#         scheduler = WarmupCosineLR(sgd, warmup_epochs=total_steps * warmup_ratio, max_epochs=total_steps)
#         print(f"Using scheduler (warmup_ratio:{warmup_ratio})")
        
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
        acc = testing()
        
        print(f"[epoch {current_epoch+1:>2d}] training loss: {Loss:.4f}, test acc: {acc:.4f} [lr: {get_lr(sgd)}]")
        if scheduler:
            scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            saves = {
                'state_dict': model.state_dict(),
                'train_dataset': checkpoint['train_dataset'],
                'test_dataset': checkpoint['test_dataset'],
                'remain_comb': comb,
            }
            savePath = modelSavePath[:-3]+ f'_finetune_best{add_str}.ckpt'
           # savePath = checkFile(savePath)
            torch.save(saves, savePath)

        if current_epoch == epoch_num-1 :
            best_acc = acc
            saves = {
                'state_dict': model.state_dict(),
                'train_dataset': checkpoint['train_dataset'],
                'test_dataset': checkpoint['test_dataset'],
                'remain_comb': comb,
            }
            savePath = modelSavePath[:-3]+ f'_finetune_last{add_str}.ckpt'
           # savePath = checkFile(savePath)
            torch.save(saves, savePath)
    return acc

total_preds = {}
def testing():
    global comb, total_preds
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

    total_preds[len(comb)] = {"predicts" : predicts, "ground_truth" : ground_truth}
    torch.save(total_preds, stdout_path.replace(".txt", "_preds.ckpt") )
    pass
    return acc

add_str = ""
if __name__=="__main__":
    try:
        from Datasets.MNIST_datasets import MNIST_Dataset
    except:
        from MNIST_datasets import MNIST_Dataset
    from Networks.LeNet import LeNet6_16_AllTanh, LeNet7_16_AllTanh, LeNet8_16_AllTanh, LeNet9_16_AllTanh
    from pretrained_models import Loads, check_rep
    import pretrained_models_CIFAR10 
    from prune_utils.prune_method import PruningMethod
    from save_stdout import SaveOutput
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type = str, default = 'MNIST4C_LeNet6_16_AllTanh_Val_fc3')
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

    parser.add_argument('--batch_size', type = int, default = 256)  
    parser.add_argument("--skip_zero", type=int, default=0)
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--rev", type=int, default=0)

    parser.add_argument("--split", type=float, default=1)

    args = parser.parse_args()
    
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
    stdout_path = stdout_path.replace('_filter_delete_order', '_finetune')
    stdout_path = stdout_path.replace('.txt', '_2.txt')
    stdout_path = check_rep(stdout_path)

    model = model.to(DEVICE)
    sys.stdout = SaveOutput(stdout_path)
    
    checkpoint   = torch.load(modelSavePath, map_location = DEVICE)
    # checkpoint   = torch.load(m_path, map_location = DEVICE)

#     checkpoint['filter_delete_order_fc3'] = []
#     breakpoint()
#     torch.save(checkpoint, modelSavePath[:-3]+f'_filter_delete_order.pt')
    
    train_dataset = checkpoint['train_dataset']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = checkpoint['val_dataset']
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = checkpoint['test_dataset']
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print()
    print(args.case.split('_')[:2])
    print(f'number of training data: {len(train_dataset)}')
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
    comb = [-1 for i in range(list(layer_initN.values())[0])]
    acc = testing()
    print(f"test acc: {acc:.4f}")

    withoutPruningAcc = acc

    for layer_name, initN in layer_initN.items():
         # 如果有指定 layer，就去看目前執行的 layer 是不是指定的，沒有就跳過
        sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero")
        if f'filter_delete_order_{sample_name}_{layer_name}' not in checkpoint.keys():
            if f'filter_delete_order_{layer_name}' not in checkpoint.keys():
                print(">>> No pruning order")
                continue
            
        add_str = "_" + layer_name
        # filter_delete_order = checkpoint[f'filter_delete_order_{layer_name}']
        sample_name = args.sampling + ( "" if args.skip_zero == 0 else "_skipZero")
        if f'filter_delete_order_{sample_name}_{layer_name}' not in checkpoint.keys():
            print(">> Try to load order which is no name")
            filter_delete_order = checkpoint[f'filter_delete_order_{layer_name}']
            print(f'>> Filter_delete_order_{layer_name}')
        else:
            filter_delete_order = checkpoint[f'filter_delete_order_{sample_name}_{layer_name}']
            print(f'>> Filter_delete_order_{sample_name}_{layer_name}')

        # filter_delete_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 13]
        # print("使用強制輸入的")
        print(f'---------- filter_delete_order_{layer_name} ----------')
        print(">>>", filter_delete_order)
        Last_accs = []
        for removedN in range(initN-layer_remaining_nodeN[layer_name], initN):
        # for removedN in range(initN-1, initN):
            comb = filter_delete_order[removedN:]
            
            model = deepcopy(backup_model).to(DEVICE)    
            for name, layer_module in model.named_modules():
                if(isinstance(layer_module, torch.nn.Linear) and name == layer_name):
                    if removedN==0:
                        print(layer_module)
                    PMethod.prune_nodes(layer_module, comb, initN)
                if(isinstance(layer_module, torch.nn.Conv2d) and name == layer_name):
                    if removedN==0:
                        print(layer_module)
                    PMethod.prune_nodes(layer_module, comb, initN)
            print(f'========== Remain {len(comb)} ==========')
            
            acc = testing()
            print(f"test acc: {acc:.4f}")
            if abs(withoutPruningAcc - acc) > 0.01:
                
                # finetune
                if "VGG" in m_path:
                    epoch_num, learning_rate = 50, 1e-2
                else:
                    epoch_num, learning_rate = 50, 1e-1
                print("learning_rate", learning_rate)
                last_acc = training(epoch_num=epoch_num, lr=learning_rate)
                Last_accs.append(round(last_acc, 4))
            else:

                Last_acc = round(acc, 4)
                # Last_accs.append(f"(w/o_finetune) {Last_acc}")
                Last_accs.append(Last_acc)
        print('finetuned acc:', Last_accs)
  #  np.save(f"prune_record/batchSize{args.batch_size}/{args.sampling}/" + args.case + "_cos_finetuned acc.npy", Last_accs)

### 同時刪除有全連接層
#     print(f'========== Without pruning ==========')
#     acc = testing()
#     print(f"test acc: {acc:.4f}")
    
#     model = deepcopy(backup_model).to(DEVICE) 
#     for layer_name, remaining_nodeN in layer_remaining_nodeN.items():
#         if f'filter_delete_order_{layer_name}' not in checkpoint.keys():
#             continue
#         filter_delete_order = checkpoint[f'filter_delete_order_{layer_name}']
#         comb = filter_delete_order[layer_initN[layer_name]-remaining_nodeN:]
               
#         for name, layer_module in model.named_modules():
#             if(isinstance(layer_module, torch.nn.Linear) and name==layer_name):
#                 PMethod.prune_nodes(layer_module, comb, layer_initN[layer_name])
#                 print(f'========== filter_delete_order_{layer_name} Remain {len(comb)} ==========')
#                 COMB = comb
    
#     acc = testing()
#     print(f"test acc: {acc:.4f}")

#     if "VGG" in m_path:
#         epoch_num, learning_rate = 50, 1e-2
#     else:
#         epoch_num, learning_rate = 50, 1e-1
#     print("learning_rate", learning_rate)
#     last_acc = training(epoch_num=epoch_num, lr=learning_rate)
#     print('finetuned acc:', last_acc)