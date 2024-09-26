from Networks.LeNet import LeNet6_16_AllTanh, LeNet7_16_AllTanh, LeNet8_16_AllTanh, LeNet9_16_AllTanh
from Networks.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, vgg8_bn
import os
from sys_util import checkkk
from datetime import datetime

def Loads(case, test=False, random=False, finetune=False, layer = "", 
          sampling = "reyei_uns", batchSize= 256, split = 1, resnet18layer = None):
    if layer != "":
        layer = "_" + layer

    add_str = '_filter_delete_order' if test else ''
    if len(add_str) == 0:
        add_str = '_finetune' if finetune else ''
    add_str2 = '_random' if random else add_str
    
    if split != 1:
        strr = f"_{split}"
    else:
        strr = ""

    current_date = datetime.now()

    # Format date as yy-mm-dd
    formatted_date = current_date.strftime("%y-%m-%d") + "/"

    if case=="CIFAR10_4C_VGG19_64_AllTanh_fc1":
        classes = [0, 4, 7, 8]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_4C_100%_validation_epoch=99-step=6299{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {0:'fc1'}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_fc1{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":128}
        layer_initN = {"fc1":128}
        
    elif case=="CIFAR10_4C_VGG19_64_AllTanh_fc2":
        classes = [0, 4, 7, 8]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_4C_100%_validation_epoch=99-step=6299{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {1:'fc2'}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":64}
        layer_initN = {"fc2":64}


    elif case=="CIFAR10_4C_VGG19_64_AllTanh":
        classes = [0, 4, 7, 8]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_4C_100%_validation_epoch=99-step=6299{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_{resnet18layer}{add_str2}.txt"
        layer_remaining_nodeN = {}
        layer_initN = {}


    elif case=="CIFAR10_10C_VGG19_64_AllTanh":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_10C_100%_validation_epoch=99-step=15699{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_{resnet18layer}{add_str2}.txt"
        layer_remaining_nodeN = {}
        layer_initN = {}
    # elif case=="CIFAR10_4C_VGG19_64_AllTanh":
    #     classes = [0, 4, 7, 8]
    #     batch_size = batchSize
    #     m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_4C_100%_validation_epoch=99-step=6299{add_str}.ckpt"
    #     model_arch = "VGG19_64_AllTanh"
    #     linear_layers = {1:'fc2', 0:"fc1"}  # forward function returns activations of [fc1, fc2]
    #     model = vgg19_bn(len(classes))
    #     stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_fc2{add_str2}.txt"
    #     layer_remaining_nodeN = {"fc1":128, "fc2":64}
    #     layer_initN = {"fc1":128, "fc2":64}

    elif case=="CIFAR10_10C_VGG19_64_AllTanh_fc1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_10C_100%_validation_epoch=99-step=15699{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {0:'fc1'}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_fc1{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":128}
        layer_initN = {"fc1":128}
        
    elif case=="CIFAR10_10C_VGG19_64_AllTanh_fc2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_10C_100%_validation_epoch=99-step=15699{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {1:'fc2'}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":64}
        layer_initN = {"fc2":64}


    elif case=="CIFAR10_10C_VGG19_64_AllTanh_conv1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_10C_100%_validation_epoch=99-step=15699{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {2:'conv1'}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_{resnet18layer}{add_str2}.txt"
        layer_remaining_nodeN = {"conv1":64}
        layer_initN = {"conv1":64}
        




    elif case=="CIFAR10_10C_VGG8_64_AllTanh_fc2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/VGG8_64_AllTanh_Val/CIFAR10_10C_100%_validation_epoch=99-step=15699{add_str}.ckpt"
        model_arch = "VGG8_64_AllTanh"
        linear_layers = {1:'fc2'}  # forward function returns activations of [fc1, fc2] 
        model = vgg8_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG8_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_fc2{add_str2}.txt"
        layer_remaining_nodeN = { "fc2":64}
        layer_initN = {"fc2":64}
    elif case=="CIFAR10_10C_VGG11_64_AllTanh_fc2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/VGG11_64_AllTanh_Val/CIFAR10_10C_100%_validation_epoch=99-step=15699{add_str}.ckpt"
        model_arch = "VGG11_64_AllTanh"
        linear_layers = {1:'fc2'}  # forward function returns activations of [fc1, fc2] 
        model = vgg11_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG11_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":64}
        layer_initN = {"fc2":64}
    elif case=="CIFAR10_10C_VGG13_64_AllTanh_fc2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/VGG13_64_AllTanh_Val/CIFAR10_10C_100%_validation_epoch=99-step=15699{add_str}.ckpt"
        model_arch = "VGG13_64_AllTanh"
        linear_layers = {1:'fc2'}  # forward function returns activations of [fc1, fc2] 
        model = vgg13_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG13_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":64}
        layer_initN = {"fc2":64}

    elif case=="CIFAR10_4C_VGG8_64_AllTanh_fc2":
        classes = [0, 4, 7, 8]
        batch_size = batchSize
        m_path = f"models/VGG8_64_AllTanh_Val/CIFAR10_4C_100%_validation_epoch=99-step=6299{add_str}.ckpt"
        model_arch = "VGG8_64_AllTanh"
        linear_layers = {1:'fc2'}  # forward function returns activations of [fc1, fc2] 
        model = vgg8_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG8_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":64}
        layer_initN = {"fc2":64}
    elif case=="CIFAR10_4C_VGG11_64_AllTanh_fc2":
        classes = [0, 4, 7, 8]
        batch_size = batchSize
        m_path = f"models/VGG11_64_AllTanh_Val/CIFAR10_4C_100%_validation_epoch=99-step=6299{add_str}.ckpt"
        model_arch = "VGG11_64_AllTanh"
        linear_layers = {1:'fc2'}  # forward function returns activations of [fc1, fc2] 
        model = vgg11_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG11_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":64}
        layer_initN = {"fc2":64}

    elif case=="CIFAR10_4C_VGG13_64_AllTanh_fc2":
        classes = [0, 4, 7, 8]
        batch_size = batchSize
        m_path = f"models/VGG13_64_AllTanh_Val/CIFAR10_4C_100%_validation_epoch=99-step=6299{add_str}.ckpt"
        model_arch = "VGG13_64_AllTanh"
        linear_layers = {1:'fc2'}  # forward function returns activations of [fc1, fc2] 
        model = vgg13_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG13_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":64}
        layer_initN = {"fc2":64}


    elif case=="CIFAR10_4C_VGG19_64_AllTanh_FC":
        classes = [0, 4, 7, 8]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_4C_100%_validation_epoch=99-step=6299{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_4C_100%_validation_epoch=99-step=6299_AllLayer_{resnet18layer}{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":128, "fc2":64}
        layer_initN = {"fc1":128, "fc2":64}

    elif case=="CIFAR10_10C_VGG19_64_AllTanh_FC":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/VGG19_64_AllTanh_Val/CIFAR10_10C_100%_validation_epoch=99-step=15699{add_str}.ckpt"
        model_arch = "VGG19_64_AllTanh"
        linear_layers = {0:'fc1', 1:'fc2'}  # forward function returns activations of [fc1, fc2]
        model = vgg19_bn(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/VGG19_64_AllTanh_Val_CIFAR10_10C_100%_validation_epoch=99-step=15699_AllLayer{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":128, "fc2":64}
        layer_initN = {"fc1":128, "fc2":64}




    # ===============================================

    elif case=="CIFAR10_4C_convNeXtAE":
        classes = [0, 4, 7, 8]
        batch_size = batchSize
        m_path = f"models/convNeXtAE/ConvNeXt_cifar4C_0.8976{add_str}.ckpt"
        model_arch = "ConvNeXtAutoEncoder_1024"
        linear_layers = {0:'encoder.head'}  # forward function returns activations of [fc1, fc2] 
        model = ConvNeXtAutoEncoder(typee = 3, neck=1024, in_chans=3)
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/ConvNeXt_cifar4C_0.8976{add_str2}.txt"
        layer_remaining_nodeN = {"encoder.head":64}
        layer_initN = {"encoder.head":64}

    elif case=="CIFAR10_10C_convNeXtAE":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/convNeXtAE/ConvNeXt_cifar10C_300_0.8908{add_str}.ckpt"
        model_arch = "ConvNeXtAutoEncoder_1024"
        linear_layers = {0:'encoder.head'}  # forward function returns activations of [fc1, fc2] 
        model = ConvNeXtAutoEncoder(typee = 3, neck=1024, in_chans=3)
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/ConvNeXt_cifar10C_300_0.8908{add_str2}.txt"
        layer_remaining_nodeN = {"encoder.head":64}
        layer_initN = {"encoder.head":64}

    elif case=="MNIST4C_convNeXtAE":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/convNeXtAE/ConvNeXt_mnist_[0, 1, 7, 8]_1_294_v0.9235_t0.9253{add_str}.ckpt"
        model_arch = "ConvNeXtAutoEncoder_1024"
        linear_layers = {0:'encoder.head'}  # forward function returns activations of [fc1, fc2] 
        model = ConvNeXtAutoEncoder(typee = 3, neck=1024, in_chans=1)
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/ConvNeXt_mnist_[0, 1, 7, 8]_1_294_v0.9235_t0.9253{add_str2}.txt"
        layer_remaining_nodeN = {"encoder.head":64}
        layer_initN = {"encoder.head":64}

    elif case=="MNIST10C_convNeXtAE":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/convNeXtAE/ConvNeXt_mnist_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_298_v0.9428_t0.9439{add_str}.ckpt"
        model_arch = "ConvNeXtAutoEncoder_1024"
        linear_layers = {0:'encoder.head'}  # forward function returns activations of [fc1, fc2] 
        model = ConvNeXtAutoEncoder(typee = 3, neck=1024, in_chans=1)
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/ConvNeXt_mnist_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_298_v0.9428_t0.9439{add_str2}.txt"
        layer_remaining_nodeN = {"encoder.head":64}
        layer_initN = {"encoder.head":64}

    else:
        assert case == 'Wrong Case'
        

    tmp = m_path.split("/")
    tmp[1] += f"/batchSize{batch_size}"
    modelSavePath = "/".join(tmp)
    checkkk("/".join(tmp[:-1]))
    checkkk(f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}")
    
    idx = 0
    if os.path.exists(stdout_path) and os.path.getsize(stdout_path) > 10:
        stdout_path = stdout_path[:-4]
        stdout_path += "_" + str(idx) + "_ver"
        while os.path.exists(stdout_path + ".txt") and os.path.getsize(stdout_path + ".txt") > 10:
            stdout_path = stdout_path.replace("_" + str(idx) + "_ver", "_" + str(idx+1) + "_ver")
            idx += 1
        stdout_path += ".txt"

    if finetune:
        return classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, layer_initN, layer_remaining_nodeN, modelSavePath
    return classes, batch_size, m_path, model_arch, linear_layers, model, stdout_path, modelSavePath
    
    