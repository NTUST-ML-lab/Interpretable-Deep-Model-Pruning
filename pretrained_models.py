from Networks.LeNet import LeNet6_16_AllTanh, LeNet7_16_AllTanh, LeNet8_16_AllTanh, LeNet9_16_AllTanh, LeNet_120_16_Tanh, LeNet5, LeNet6_16_AllTanh_bf_act, LeNet5_128_64_tanh
from Networks.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from Networks.vgg_128 import vgg19_bn_128
import os
from sys_util import checkkk
from datetime import datetime


# 0:"fc1" 1:"fc2" 2:"fc3" 3:"conv1" 4:"conv2"

def Loads(case, test=False, random=False, layer = "", finetune=False,
           sampling = "reyei_uns", batchSize = 256, explain = False, split = "1", resnet18layer = None, remove = False):
    if layer != "":
        layer = "_" + layer

    add_str = '_filter_delete_order' if test else ''
    if len(add_str) == 0:
        add_str = '_finetune' if finetune else ''
    add_str2 = '_random' if random else add_str
    save_str = "_remove" if remove else ""
    

    if (split != "1.0" and split != "1") and len(split) > 0:
        strr = f"_{split}"
    else:
        strr = ""
        

    # Get current date
    current_date = datetime.now()

    # Format date as yy-mm-dd
    formatted_date = current_date.strftime("%y-%m-%d") + "/"


    # MNIST 4-class LeNet6-16 AllTanh validation =====================================================

    if case=="MNIST4C_LeNet6_16_AllTanh_Val_fc3":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist4C_100.0%_0.0270{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 2:"fc3"}  # forward function returns activations of [fc3]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_fc3{add_str2}.txt"
        layer_remaining_nodeN = {"fc3":16}
        layer_initN = {"fc3":16}

    elif case=="MNIST4C_LeNet6_16_AllTanh_Val_fc2":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist4C_100.0%_0.0270{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 1:"fc2"}  # forward function returns activations of [fc2]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":84}
        layer_initN = { "fc2":84}

    elif case=="MNIST4C_LeNet6_16_AllTanh_Val_fc1":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist4C_100.0%_0.0270{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 0:"fc1"}  # forward function returns activations of [fc1]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_fc1{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":120}
        layer_initN = {"fc1":120}

    elif case=="MNIST4C_LeNet6_16_AllTanh_Val_conv2":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist4C_100.0%_0.0270{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 4:"conv2"}  # forward function returns activations of [conv2]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_conv2{add_str2}.txt"
        layer_remaining_nodeN = {"conv2":16}
        layer_initN = { "conv2":16}

    elif case=="MNIST4C_LeNet6_16_AllTanh_Val_conv1":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist4C_100.0%_0.0270{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 3:"conv1"}  # forward function returns activations of [conv1]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_conv1{add_str2}.txt"
        layer_remaining_nodeN = {"conv1":6}
        layer_initN = { "conv1":6}

    elif case=="MNIST4C_LeNet6_16_AllTanh_Val_FC":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist4C_100.0%_0.0270{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 0:"fc1", 1:"fc2", 2:"fc3"}  # forward function returns activations of [conv1]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_AllLayer{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":120, "fc2":84, "fc3":16}
        layer_initN = {"fc1":120, "fc2":84, "fc3":16}

    # =======================================================================================

    # MNIST 10-class LeNet6-16 AllTanh validation =====================================================

    elif case=="MNIST10C_LeNet6_16_AllTanh_Val_fc3":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist10C_100.0%_0.0928{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 2:"fc3"}  # forward function returns activations of [fc3]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_fc3{add_str2}.txt"
        layer_remaining_nodeN = {"fc3":16}
        layer_initN = {"fc3":16}

    elif case=="MNIST10C_LeNet6_16_AllTanh_Val_fc2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist10C_100.0%_0.0928{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 1:"fc2"}  # forward function returns activations of [fc2]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":84}
        layer_initN = {"fc2":84}

    elif case=="MNIST10C_LeNet6_16_AllTanh_Val_fc1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist10C_100.0%_0.0928{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 0:"fc1"}  # forward function returns activations of [fc1]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_fc1{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":120}
        layer_initN = {"fc1":120}

    elif case=="MNIST10C_LeNet6_16_AllTanh_Val_conv2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist10C_100.0%_0.0928{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 4:"conv2"}  # forward function returns activations of [conv2]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_conv2{add_str2}.txt"
        layer_remaining_nodeN = {"conv2":16}
        layer_initN = {"conv2":16}

    elif case=="MNIST10C_LeNet6_16_AllTanh_Val_conv1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist10C_100.0%_0.0928{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 3:"conv1"}  # forward function returns activations of [conv1]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_conv1{add_str2}.txt"
        layer_remaining_nodeN = {"conv1":6}
        layer_initN = {"conv1":6}

    elif case=="MNIST10C_LeNet6_16_AllTanh_Val":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist10C_100.0%_0.0928{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 3:"conv1", 4:"conv2", 0:"fc1", 1:"fc2", 2:"fc3"}  # forward function returns activations of [conv1]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_AllLayer{add_str2}.txt"
        layer_remaining_nodeN = {"conv1":6, "conv2":16, "fc1":120, "fc2":84, "fc3":16}
        layer_initN = {"conv1":6, "conv2":16, "fc1":120, "fc2":84, "fc3":16}

    elif case=="MNIST4C_LeNet6_16_AllTanh_Val":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist4C_100.0%_0.0270{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 3:"conv1", 4:"conv2", 0:"fc1", 1:"fc2", 2:"fc3"}  # forward function returns activations of [conv1]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist4C_100.0%_0.0270_AllLayer{add_str2}.txt"
        layer_remaining_nodeN = {"conv1":6, "conv2":16, "fc1":120, "fc2":84, "fc3":16}
        layer_initN = {"conv1":6, "conv2":16, "fc1":120, "fc2":84, "fc3":16}

    elif case=="MNIST10C_LeNet6_16_AllTanh_Val_FC":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val/mnist10C_100.0%_0.0928{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh"
        linear_layers = { 0:"fc1", 1:"fc2", 2:"fc3"}  # forward function returns activations of [conv1]
        model = LeNet6_16_AllTanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_mnist10C_100.0%_0.0928_AllLayer{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":120, "fc2":84, "fc3":16}
        layer_initN = {"fc1":120, "fc2":84, "fc3":16}




    elif case=="MNIST10C_LeNet6_16_AllTanh_Val_bfact_conv1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet6_16_AllTanh_Val_bfact/mnist10C_100.0%_0.0928{add_str}.pt"
        model_arch = "LeNet6_16_AllTanh_Val_bfact"
        linear_layers = { 3:"conv1"}  # forward function returns activations of [conv1]
        model = LeNet6_16_AllTanh_bf_act(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet6_16_AllTanh_Val_bfact_mnist10C_100.0%_0.0928_conv1{add_str2}.txt"
        layer_remaining_nodeN = {"conv1":6}
        layer_initN = {"conv1":6}
    # =======================================================================================

    elif case=="MNIST4C_AE_50":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/AE/MNIST4C/mnist_4C_100.0%_1005_50_0.0000{add_str}.pt"
        model_arch = "Deep_AE_50"
        linear_layers = { 0:"fc3"} 
        model = Deep_AE_50()
        stdout_path = f"prune_record/{formatted_date}AE/batchSize{batch_size}{strr}/{sampling}/mnist_4C_100.0%_1005_50_0.0000{add_str2}.txt"
        layer_remaining_nodeN = {"fc3":50}
        layer_initN = {"fc3":50}

    elif case=="MNIST10C_AE_50":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/AE/MNIST10C/mnist_10C_100.0%_405_50_0.0000{add_str}.pt"
        model_arch = "Deep_AE_50"
        linear_layers = { 0:"fc3"} 
        model = Deep_AE_50()
        stdout_path = f"prune_record/{formatted_date}AE/batchSize{batch_size}{strr}/{sampling}/mnist_10C_100.0%_405_50_0.0000{add_str2}.txt"
        layer_remaining_nodeN = {"fc3":50}
        layer_initN = {"fc3":50}

    elif case=="MNIST4C_AE_16":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/AE/MNIST4C/mnist_4C_100.0%_405_16_0.0001{add_str}.pt"
        model_arch = "Deep_AE_16"
        linear_layers = { 0:"fc3"} 
        model = Deep_AE_16()
        stdout_path = f"prune_record/{formatted_date}AE/batchSize{batch_size}{strr}/{sampling}/mnist_4C_100.0%_405_16_0.0001{add_str2}.txt"
        layer_remaining_nodeN = {"fc3":16}
        layer_initN = {"fc3":16}

    elif case=="MNIST10C_AE_16":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/AE/MNIST10C/mnist_10C_100.0%_405_16_0.0000{add_str}.pt"
        model_arch = "Deep_AE_16"
        linear_layers = { 0:"fc3"} 
        model = Deep_AE_16()
        stdout_path = f"prune_record/{formatted_date}AE/batchSize{batch_size}{strr}/{sampling}/mnist_10C_100.0%_405_16_0.0000{add_str2}.txt"
        layer_remaining_nodeN = {"fc3":16}
        layer_initN = {"fc3":16}


    # ============================================================================================

    elif case=="MNIST4C_SAE_16":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/SAE/MNIST4C/mnist_4C_100.0%_405_0.0150{add_str}.pt"
        model_arch = "Deep_SAE_16"
        linear_layers = { 0:"fc3"} 
        model = Deep_SAE_16(classnum=len(classes))
        stdout_path = f"prune_record/{formatted_date}SAE/batchSize{batch_size}{strr}/{sampling}/mnist_4C_100.0%_405_0.0150{add_str2}.txt"
        layer_remaining_nodeN = {"fc3":16}
        layer_initN = {"fc3":16}

    elif case=="MNIST10C_SAE_16":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/SAE/MNIST10C/mnist_10C_100.0%_405_0.1191{add_str}.pt"
        model_arch = "Deep_SAE_16"
        linear_layers = { 0:"fc3"} 
        model = Deep_SAE_16(classnum=len(classes))
        stdout_path = f"prune_record/{formatted_date}SAE/batchSize{batch_size}{strr}/{sampling}/mnist_10C_100.0%_405_0.1191{add_str2}.txt"
        layer_remaining_nodeN = {"fc3":16}
        layer_initN = {"fc3":16}


    # =======================================================================================

    elif case=="MNIST4C_DAE_16":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/DAE/MNIST4C/mnist_4C_100.0%_best_val{add_str}.ckpt"
        model_arch = "Deep_DAE_16"
        linear_layers = { 0:'second_autoencoder.encoder.encoder.4'} 
        model = get_DAE(class_num=len(classes))
        stdout_path = f"prune_record/{formatted_date}DAE/batchSize{batch_size}{strr}/{sampling}/mnist_4C_100.0%_best_val{add_str2}.txt"
        layer_remaining_nodeN = {'second_autoencoder.encoder.encoder.4':16}
        layer_initN = {'second_autoencoder.encoder.encoder.4':16}

    elif case=="MNIST10C_DAE_16":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/DAE/MNIST10C/mnist_10C_100.0%_best_val{add_str}.ckpt"
        model_arch = "DCAE_NCLS6_16"
        linear_layers = { 0:'second_autoencoder.encoder.encoder.6'} 
        model = get_DAE(class_num=len(classes))
        stdout_path = f"prune_record/{formatted_date}DAE/batchSize{batch_size}{strr}/{sampling}/mnist_10C_100.0%_best_val{add_str2}.txt"
        layer_remaining_nodeN = {'second_autoencoder.encoder.encoder.6':16}
        layer_initN = {'second_autoencoder.encoder.encoder.6':16}



    # =======================================================================================

    elif case=="MNIST4C_IC_16":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/IC/MNIST4C/mnist4C_16_iter_75000{add_str}.ckpt"
        model_arch = "IC"
        linear_layers = { 0:'Encoder.fc'} 
        model = ImageCompressor()
        stdout_path = f"prune_record/{formatted_date}IC/batchSize{batch_size}{strr}/{sampling}/mnist4C_16_iter_75000{add_str2}.txt"
        layer_remaining_nodeN = {'Encoder.fc':16}
        layer_initN = {'Encoder.fc':16}

    elif case=="MNIST10C_IC_16":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        batch_size = batchSize
        m_path = f"models/IC/MNIST10C/mnist10C_16_iter_75012{add_str}.ckpt"
        model_arch = "IC"
        linear_layers = { 0:'Encoder.fc'} 
        model = ImageCompressor()
        stdout_path = f"prune_record/{formatted_date}IC/batchSize{batch_size}{strr}/{sampling}/mnist10C_16_iter_75012{add_str2}.txt"
        layer_remaining_nodeN = {'Encoder.fc':16}
        layer_initN = {'Encoder.fc':16}

    elif case=="MNIST10C_IC_32":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        batch_size = batchSize
        m_path = f"models/IC/MNIST10C/mnist10C_32_iter_350056{add_str}.ckpt"
        model_arch = "IC"
        linear_layers = { 0:'Encoder.fc'} 
        model = ImageCompressor(neck=32)
        stdout_path = f"prune_record/{formatted_date}IC/batchSize{batch_size}{strr}/{sampling}/mnist10C_32_iter_350056{add_str2}.txt"
        layer_remaining_nodeN = {'Encoder.fc':32}
        layer_initN = {'Encoder.fc':32}

    elif case=="MNIST10C_IC_32_Tanh":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        batch_size = batchSize
        m_path = f"models/IC_Tanh/MNIST10C/mnist10C_32_iter_300048{add_str}.ckpt"
        model_arch = "IC"
        linear_layers = { 0:'Encoder.fc'} 
        model = ImageCompressor(neck=32, ifTanh = True)
        stdout_path = f"prune_record/{formatted_date}IC_Tanh/batchSize{batch_size}{strr}/{sampling}/mnist10C_32_iter_300048{add_str2}.txt"
        layer_remaining_nodeN = {'Encoder.fc':32}
        layer_initN = {'Encoder.fc':32}
    # =======================================================================================

    # =======================================================================================

    elif case=="MNIST10C_DCAE_NCLS6_16":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/DCAE_NCLS6_16/MNIST10C/mnist_10C_100.0%_[64,64]1024,1024,256,64,16_0.0150_0.9099{add_str}.ckpt"
        model_arch = "DCAE_NCLS6_16"
        linear_layers = { 0:'second_autoencoder.encoder.encoder.6'} 
        model = get_DCAE_nocls(dataset = "mnist", class_num=len(classes), neck=16, net="1024,1024,256,64,", convlayer=[64,64], typee=1)
        stdout_path = f"prune_record/{formatted_date}DCAE_NCLS6_16/batchSize{batch_size}{strr}/{sampling}/mnist_10C_100.0%_[64,64]1024,1024,256,64,16_0.0150_0.9099{add_str2}.txt"
        layer_remaining_nodeN = {'second_autoencoder.encoder.encoder.6':16}
        layer_initN = {'second_autoencoder.encoder.encoder.6':16}

    elif case=="MNIST4C_DCAE_NCLS6_16":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/DCAE_NCLS6_16/MNIST4C/mnist_4C_100.0%_[64,64]1024,1024,256,64,16_0.0153_0.9127{add_str}.ckpt"
        model_arch = "DCAE_NCLS6_16"
        linear_layers = { 0:'second_autoencoder.encoder.encoder.6'} 
        model = get_DCAE_nocls(dataset = "mnist", class_num=len(classes), neck=16, net="1024,1024,256,64,", convlayer=[64,64], typee=1)
        stdout_path = f"prune_record/{formatted_date}DCAE_NCLS6_16/batchSize{batch_size}{strr}/{sampling}/mnist_4C_100.0%_[64,64]1024,1024,256,64,16_0.0153_0.9127{add_str2}.txt"
        layer_remaining_nodeN = {'second_autoencoder.encoder.encoder.6':16}
        layer_initN = {'second_autoencoder.encoder.encoder.6':16}

    # =======================================================================================
    elif case=="MNIST4C_convNeXtAE":
        classes = [0, 1, 7, 8]
        batch_size = batchSize
        m_path = f"models/convNeXtAE/convNeXt_mnist_[0, 1, 7, 8]_297_v0.9125_t0.9153{add_str}.ckpt"
        model_arch = "ConvNeXtAutoEncoder_1024"
        linear_layers = {0:'encoder.head'}  # forward function returns activations of [fc1, fc2] 
        model = ConvNeXtAutoEncoder(typee = 3, neck=1024, in_chans=1)
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/convNeXt_mnist_[0, 1, 7, 8]_297_v0.9125_t0.9153{add_str2}.txt"
        layer_remaining_nodeN = {"encoder.head":64}
        layer_initN = {"encoder.head":64}

    elif case=="MNIST10C_convNeXtAE":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/convNeXtAE/convNeXt_mnist_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_298_v0.9428_t0.9439{add_str}.ckpt"
        model_arch = "ConvNeXtAutoEncoder_1024"
        linear_layers = {0:'encoder.head'}  # forward function returns activations of [fc1, fc2] 
        model = ConvNeXtAutoEncoder(typee = 3, neck=1024, in_chans=1)
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/convNeXt_mnist_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_298_v0.9428_t0.9439{add_str2}.txt"
        layer_remaining_nodeN = {"encoder.head":64}
        layer_initN = {"encoder.head":64}
    # =======================================================================================
    elif case=="MNIST10C_LeNet5_UFKT_conv1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet5_UKFT/base{add_str}.pt"
        model_arch = "LeNet5"
        linear_layers = { 0:"conv1"}  # forward function returns activations of [conv2]
        model = LeNet5(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet5_UKFT_conv1{add_str2}.txt"
        layer_remaining_nodeN = {"conv1":20}
        layer_initN = {"conv1":20}
    elif case=="MNIST10C_LeNet5_UFKT_conv2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet5_UKFT/base{add_str}.pt"
        model_arch = "LeNet5"
        linear_layers = { 1:"conv2"}  # forward function returns activations of [conv2]
        model = LeNet5(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet5_UKFT_conv2{add_str2}.txt"
        layer_remaining_nodeN = {"conv2":50}
        layer_initN = {"conv2":50}

    elif case=="MNIST10C_LeNet5_UFKT_fc1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet5_UKFT/base{add_str}.pt"
        model_arch = "LeNet5"
        linear_layers = { 2:"fc1"}  # forward function returns activations of [conv2]
        model = LeNet5(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet5_UKFT_fc1{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":800}
        layer_initN = {"fc1":800}

    elif case=="MNIST10C_LeNet5_UFKT_fc2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet5_UKFT/base{add_str}.pt"
        model_arch = "LeNet5"
        linear_layers = { 3:"fc2"}  # forward function returns activations of [conv2]
        model = LeNet5(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet5_UKFT_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":500}
        layer_initN = {"fc2":500}

    # =======================================================================================
    elif case=="MNIST10C_LeNet5_128_64_AllTanh_conv1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet5_128_64_AllTanh/mnist10C_100.0%_tanh_99_0.0488{add_str}.pt"
        model_arch = "LeNet5_128_64_AllTanh"
        linear_layers = { 0:"conv1"}  # forward function returns activations of [conv2]
        model = LeNet5_128_64_tanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet5_128_64_AllTanh_mnist10C_100.0%_tanh_0.0488_conv1{add_str2}.txt"
        layer_remaining_nodeN = {"conv1":20}
        layer_initN = {"conv1":20}
    elif case=="MNIST10C_LeNet5_128_64_AllTanh_conv2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet5_128_64_AllTanh/mnist10C_100.0%_tanh_99_0.0488{add_str}.pt"
        model_arch = "LeNet5_128_64_AllTanh"
        linear_layers = { 1:"conv2"}  # forward function returns activations of [conv2]
        model = LeNet5_128_64_tanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet5_128_64_AllTanh_mnist10C_100.0%_tanh_0.0488_conv2{add_str2}.txt"
        layer_remaining_nodeN = {"conv2":50}
        layer_initN = {"conv2":50}

    elif case=="MNIST10C_LeNet5_128_64_AllTanh_fc1":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet5_128_64_AllTanh/mnist10C_100.0%_tanh_99_0.0488{add_str}.pt"
        model_arch = "LeNet5_128_64_AllTanh"
        linear_layers = { 2:"fc1"}  # forward function returns activations of [conv2]
        model = LeNet5_128_64_tanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet5_128_64_AllTanh_mnist10C_100.0%_tanh_0.0488_fc1{add_str2}.txt"
        layer_remaining_nodeN = {"fc1":128}
        layer_initN = {"fc1":128}

    elif case=="MNIST10C_LeNet5_128_64_AllTanh_fc2":
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size = batchSize
        m_path = f"models/LeNet5_128_64_AllTanh/mnist10C_100.0%_tanh_99_0.0488{add_str}.pt"
        model_arch = "LeNet5_128_64_AllTanh"
        linear_layers = { 3:"fc2"}  # forward function returns activations of [conv2]
        model = LeNet5_128_64_tanh(len(classes))
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/LeNet5_128_64_AllTanh_mnist10C_100.0%_tanh_0.0488_fc2{add_str2}.txt"
        layer_remaining_nodeN = {"fc2":64}
        layer_initN = {"fc2":64}

    # =====

    elif case=="Mini_ImageNet_resnet18_relu":
        from Networks.model_center import get_model
        class tmp():
            def __init__(self) -> None:
                self.arch = "resnet18"
                self.activate = "relu"
                self.pretrained = False
                self.datasets = "mini_imageNet"

        classes = [i for i in range(100)]
        batch_size = batchSize
        m_path = f"models/ResNet18_AllReLU/MiniImageNet_resnet_allReLU_199_include10%Data{add_str}.ckpt"
        model_arch = "ResNet18_AllReLU"
        linear_layers = { }  # forward function returns activations of [conv2]
        model = get_model(tmp())
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/MiniImageNet_resnet_allReLU_199_include10%Data_{resnet18layer}{add_str2}{save_str}.txt"
        layer_remaining_nodeN = {}
        layer_initN = {
            "1" :64, "2":128, "3":256, "4":512
        }

    elif case=="Mini_ImageNet_resnet18_relu_ES":
        from Networks.model_center import get_model
        class tmp():
            def __init__(self) -> None:
                self.arch = "resnet18"
                self.activate = "relu"
                self.pretrained = False
                self.datasets = "mini_imageNet"

        classes = [i for i in range(100)]
        batch_size = batchSize
        m_path = f"models/ResNet18_AllReLU_ES/MiniImageNet_resnet_allReLU_188ES_include10%Data_val{add_str}.ckpt"
        model_arch = "ResNet18_AllReLU_ES"
        linear_layers = { }  # forward function returns activations of [conv2]
        model = get_model(tmp())
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/MiniImageNet_resnet_allReLU_188ES_include10%Data_val_{resnet18layer}{add_str2}{save_str}.txt"
        layer_remaining_nodeN = {}
        layer_initN = {}

    elif case=="Mini_ImageNet_resnet18_tanh":
        from Networks.model_center import get_model
        class tmp():
            def __init__(self) -> None:
                self.arch = "resnet18"
                self.activate = "tanh"
                self.pretrained = False
                self.datasets = "mini_imageNet"

        classes = [i for i in range(100)]
        batch_size = batchSize
        m_path = f"models/ResNet18_AllTanh/MiniImageNet_resnet_AllTanh_199_include10%Data{add_str}.ckpt"
        model_arch = "ResNet18_AllTanh"
        linear_layers = { }  # forward function returns activations of [conv2]
        model = get_model(tmp())
        stdout_path = f"prune_record/{formatted_date}batchSize{batch_size}{strr}/{sampling}/MiniImageNet_resnet_AllTanh_199_include10%Data_{resnet18layer}{add_str2}{save_str}.txt"
        layer_remaining_nodeN = {}
        layer_initN = {}



    else:
        assert case == 'Wrong Case'

    tmp = m_path.split("/")

    tmp[1] += f"/batchSize{batch_size}{strr}"
    if explain:
        tmp[1] += f"/explain"
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
    
def check_rep(stdout_path):
    idx = 0
    if os.path.exists(stdout_path) and os.path.getsize(stdout_path) > 10:
        stdout_path = stdout_path[:-4]
        stdout_path += "_" + str(idx) + "_ver"
        while os.path.exists(stdout_path + ".txt") and os.path.getsize(stdout_path + ".txt") > 10:
            stdout_path = stdout_path.replace("_" + str(idx) + "_ver", "_" + str(idx+1) + "_ver")
            idx += 1
        stdout_path += ".txt"
    return stdout_path