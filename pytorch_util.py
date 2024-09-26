from torch.utils.data import TensorDataset
import torch
import numpy as np
import random
import pandas as pd


class Logger:
    def __init__(self):
        self.train_loss = []
        self.testing_loss = []
        self.train_accuracy = []
        self.testing_accuracy = []
        self.row = []

    def reset(self):
        del self.train_loss
        del self.testing_loss
        del self.train_accuracy
        del self.testing_accuracy
        self.train_loss = []
        self.testing_loss = []
        self.train_accuracy = []
        self.testing_accuracy = []

    def addRecord(self, train_loss = "", train_acc = "", testing_loss = "", testing_acc="", message = ""):
        if type(train_loss) != str or len(train_loss) != 0:
            self.train_loss.append(train_loss)
        if type(train_acc) != str or len(train_acc) != 0:
            self.train_accuracy.append(train_acc)
        if type(testing_loss) != str or len(testing_loss) != 0:
            self.testing_loss.append(testing_loss)
        if type(testing_acc) != str or len(testing_acc) != 0:
            self.testing_accuracy.append(testing_acc)
        if type(testing_acc) != str or len(testing_acc) != 0:
            self.row.append(message)

    def save(self, path):
        with open(path, "w") as f:
            f.write(",train_loss,testing_loss,train_accuracy,testing_accuracy\n")
            maxx = max([len(self.train_loss), len(self.train_accuracy), len(self.testing_loss), len(self.testing_accuracy)] )

            for i in range(maxx):
                try:
                    f.write( str(self.row[i]) + "," )
                except:
                    f.write(",")
                try:
                    f.write( str(self.train_loss[i]) + "," )
                except:
                    f.write(",")
                try:
                    f.write( str(self.testing_loss[i]) + "," )
                except:
                    f.write(",")
                try:
                    f.write( str(self.train_accuracy[i]) + "," )
                except:
                    f.write(",")
                try:
                    f.write( str(self.testing_accuracy[i]) + "\n" )
                except:
                    f.write("\n")

        return

def getDevice(gpu = "cuda:0"):
    return torch.device( gpu if torch.cuda.is_available() else "cpu")

try:
    from prettytable import PrettyTable
    def count_parameters(model, printTable = False):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            if printTable:
                table.add_row([name, params])
            total_params+=params
        if printTable:
            print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
except:
    pass
try:
    from thop import profile
    def count_target_ops(model, example_inputs, targets = [], verbose = False):
        macs, params, ret_dict = profile(model, (example_inputs, ), ret_layer_info=True, verbose=verbose)
        if len(targets) != 0: 
            macs = 0
            params = 0
            for key, value in ret_dict.items():
                if key in targets:
                    macs += value[0]
                    params += value[1]
        return macs, params
except:
    pass
    
def setup_seed(seed, deterministic = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic

def printModelStatus(model, typee = 0, target = "", show = True, gather = False, ifPrint = True):
    '''
    type 0: all of weight
    type 1: max, min, avg, all zero count
    '''
    ret = []

    if typee == 0:
        for param in model.named_parameters():
            print(param[0], param[1])
    if typee in [1, 2, 3]:
        maxx, minn, avg, az = 1e-10, 1e10, 0, 0
        if show:
            if typee == 3 :
                if ifPrint:
                    print('%30s %20s' % ("Name", "Shape") )
            else:
                if ifPrint:
                    print('%30s %10s %10s %10s %15s %15s' % ("Name", "Max", "Min", "Mean", "pruned count", "total count") )
        for param in model.named_parameters():
            if len(target) == 0 or target == param[0]:

                maxx = float(param[1].max().data.detach().cpu())
                minn = float(param[1].min())
                avg = float(param[1].mean())
                for node in list(param[1].data.detach().cpu()):
                    if node.max() == 0 and  node.min() == 0:
                        az += 1
                if typee <= 2 and ifPrint:
                    print('%30s %10s %10s %10s %15s %15s' % ( str(param[0]), str(maxx)[:6], str(minn)[:6], str(avg)[:6], str(az), str(param[1].size(0)) ) )
                if typee == 3:
                    if ifPrint:
                        print('%30s %20s' % (str(param[0]), param[1].shape) )
                    if gather:
                        ret = {str(param[0]) : param[1].shape} 
                if typee == 2 and ifPrint:
                    print(param[0], param[1])
                if target == param[0]:
                    break

    return ret


def dw(data:torch.tensor, typee = "numpy"):
    try:
        if typee == "numpy":
            return data.data.detach().cpu().numpy()
        else:
            return data.data.detach().cpu().tolist()
    except:
        if typee == "numpy":
            return data.data.detach().numpy()
        else:
            return data.data.detach().tolist()
        

def csv_to_dataset(path, label = "", label_type = "long", skip = []):
    x, y = [], []
    df = pd.read_csv(path)
    column = df.columns
    for c in column:
        if not (c in skip or c == label):
            x.append(df[c].to_numpy())
    x = torch.tensor(np.array(x).transpose())
    if len("label") != 0:
        if label_type.lower() == "long":
            y = df[label].to_numpy()
            tmp = np.zeros( (len(y), max(y)+1) )
            for i, l in enumerate(y):
                tmp[i][l] = 1
            y = torch.LongTensor(tmp)
        else:
            y = torch.FloatTensor(df[label].to_numpy())
    else:
        return TensorDataset(x)
    return TensorDataset(x, y)





def one_hot( y, gpu = True):

    try:
        y = torch.from_numpy(y)
    except TypeError:
        None

    y_1d = y
    if gpu:
        y_hot = torch.zeros((y.size(0), torch.max(y).int()+1)).cuda()
    else:
        y_hot = torch.zeros((y.size(0), torch.max(y).int()+1))

    for i in range(y.size(0)):
        y_hot[i, y_1d[i].int()] = 1

    return y_hot