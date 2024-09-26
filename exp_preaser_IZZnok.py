import os
from pretrained_models import Loads
from sys_util import checkkk
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


print("")



def to_list(TEXT, target):
    L = TEXT.split('\n')
    rows = []
    pattern = r"[-+]?(?:\d*\.*\d+)"
    for i, t in enumerate(L):
        try:
            n, acc, tc, Ixz, Izy = re.findall(pattern, t)
            D = {'acc': acc, 'tc': tc, 'Ixz': Ixz, 'Izy': Izy}
        except:
            try:
                n, acc, tc = re.findall(pattern, t)
                D = {'acc': acc, 'tc': tc}
            except:
                n, acc, tc, Ixz, Izy, IZZnok = re.findall(pattern, t)
                D = {'acc': acc, 'tc': tc, 'Ixz': Ixz, 'Izy': Izy, 'IZZnok':IZZnok }
        rows.append(float(D[target]))
    return rows

def to_dict(TEXT, target):
    L = TEXT.split('\n')
    rows = []
    pattern = r"[-+]?(?:\d*\.*\d+)"
    for i, t in enumerate(L):
#         if "remain" in t:
#             remain_num = re.findall(pattern, t)[0]
#         if "[epoch 10]" not in t:
#             continue
#         n, loss, acc = re.findall(pattern, t)
#         rows.append(f'{remain_num} & {acc}')

        try:
            n, acc, tc, Ixz, Izy = re.findall(pattern, t)
            D = {'acc': acc, 'tc': tc, 'Ixz': Ixz, 'Izy': Izy}
        except:
            try:
                n, acc, tc = re.findall(pattern, t)
                D = {'acc': acc, 'tc': tc}
            except:
                n, acc, tc, Ixz, Izy, IZZnok = re.findall(pattern, t)
                D = {'acc': acc, 'tc': tc, 'Ixz': Ixz, 'Izy': Izy, 'IZZnok':IZZnok}
        rows.append(f'{n} & {D[target]}')
    return '\n'.join(rows)


# classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, modelSavePath = Loads(case_name)


# pretrain_name = modelSavePath[7:-5].replace("/", "_")
'''
renyi_IZY
renyi_Hz
reyei_uns
renyi_uns


MiniImageNet_resnet_allReLU_199_include10%Data_.layer4.1.activation
MiniImageNet_resnet_allReLU_199_include10%Data_block1
'''

method = "bin"
case_name = "VGG19_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_fc1"
date = "24-06-22"
# case_name = "CIFAR10_10C_VGG19_128_AllTanh"
batch = "batchSize256"
addstr = ""

batch = batch.replace("batchSize", "")



pretrain_name = (f"{method}/{case_name}"   )
toks = pretrain_name.split("_")
dataset = toks[4] + "_" + toks[5]


if len(date) != 0:
    date = "/" + date

save_df = checkkk(f"prune_record{date}/batchSize{batch}/formatted")
result_df = f"prune_record{date}/batchSize{batch}"

'''
case = "LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_fc1"
case = "LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_fc2"
case = "LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_fc3"
case = "LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_fc3"
case = "LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_fc2"
case = "VGG19_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_fc2"
case = "VGG19_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_fc1"
case = "VGG19_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_fc2"


LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_conv2
LeNet5_128_64_AllTanh_mnist10C_100.0%_tanh_0.0488_conv1
_mnist10C_100.0%_tanh_0.0488

'''


file_name = [result_df + "/" + pretrain_name + f"_filter_delete_order{addstr}.txt", result_df + "/" + pretrain_name + "_random.txt"]
target = "-----------"
target2 = "[remain"


contains = []
dictt = {}
filters = []
informationPlane = []
randoms = []
checkkk(f"prune_record{date}/batchSize{batch}/formatted/{method}")
w_f = open( save_df + "/" + pretrain_name + f"{addstr}_parsered.txt", "w")
w_f.write("# " + pretrain_name + "\n")
w_f.write("dataset = \'" + toks[3] + "_" + toks[4] + "\'\n\n\n")

for fn in file_name:
    try:
        with open(fn, "r") as f:
            
            f = list(f)

            parsetItem = ""
            flag = False
            for idx, line in enumerate(f):
                if target in line:
                    parsetItem = ""
                    layer = line.split(" ")[1]
                    if "random" not in fn:
                        dictt[layer] = len(filters)
                        title = "f\"{model}_" + layer + "_" + f[idx+3].split("]")[0][8:-5] + "_{dataset}_100%\""
                        contains.append( "title = " + title )

                    for l in range(idx+1, len(f)):
                        if target in f[l]:
                            break
                        if target2 in f[l]:
                            parsetItem += f[l]
                    if "random" not in fn:
                        filters.append("Filter_renyi = \\\n\"")
                        filters[-1] += (   to_dict(parsetItem[:-1], 'acc').replace("\n", "\\n") + "\"\n")
                        filters[-1] += ("Filter_renyi_TC = \\\n\"")
                        filters[-1] += (   to_dict(parsetItem[:-1], 'tc').replace("\n", "\\n") + "\"\n")
                        filters[-1] += ("Filter_renyi_IXZ = \\\n\"")
                        filters[-1] += (   to_dict(parsetItem[:-1], 'Ixz').replace("\n", "\\n") + "\"\n")
                        filters[-1] += ("Filter_renyi_IZY = \\\n\"")
                        filters[-1] += (   to_dict(parsetItem[:-1], 'Izy').replace("\n", "\\n") + "\"\n")
                        filters[-1] += ("Filter_renyi_IZZnok = \\\n\"")
                        filters[-1] += (   to_dict(parsetItem[:-1], 'IZZnok').replace("\n", "\\n") + "\"\n")
                        informationPlane.append("Ixz = ")
                        informationPlane[-1] += str(to_list(parsetItem[:-1], 'Ixz'))
                        informationPlane[-1] += ("\nIzy = ")
                        informationPlane[-1] += str(to_list(parsetItem[:-1], 'Izy'))
                        informationPlane[-1] += ("\nIZZnok = ")
                        informationPlane[-1] += str(to_list(parsetItem[:-1], 'IZZnok'))
                        informationPlane[-1] += ("\n")
                    else:
                        randoms.append("Random = \\\n\"")
                        randoms[-1] += (filters[dictt[layer]].split("Filter_renyi = ")[1].split("\\n")[0][3:] + "\\n")
                        randoms[-1] += (   to_dict(parsetItem[:-1], 'acc').replace("\n", "\\n") + "\"\n")
                        randoms[-1] += ("Random_TC = \\\n\"")
                        randoms[-1] += (filters[dictt[layer]].split("Filter_renyi_TC = ")[1].split("\\n")[0][3:] + "\\n")
                        randoms[-1] += (   to_dict(parsetItem[:-1], 'tc').replace("\n", "\\n") + "\"\n")
                    pass

    except Exception as e:
        print(e)
        continue        
for i in range(len(contains)):
    w_f.write(contains[i] + "\n")
    w_f.write(filters[i] + "\n")
    try:
        w_f.write(randoms[i] + "\n\n\n")
    except:
        continue

w_f.write("\n\nInformation plane\n\n")
for i in range(len(contains)):
    w_f.write("dataset = \'" + toks[3] + "_" + toks[4] + "\'\n")
    w_f.write(contains[i] + "\n")
    w_f.write(informationPlane[i] + "\n")

w_f.close()


