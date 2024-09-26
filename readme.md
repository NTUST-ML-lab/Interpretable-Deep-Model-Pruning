## This is the official code for Interpretable Deep Model Pruning
### Checkpoint
- As for the model checkpoint, you can download at: https://1drv.ms/u/s!Ao-9hEYwmdRi9zQZC22bpL90Ck8O?e=nJKIhb
- The checkpoint needs to include the following keys:
1. state_dict: The state_dict of the model
2. test_dataset: The test dataset corresponding to the dataset
3. For the MNIST dataset, the following additional keys need to be included:
   1. train_dataset: The MNIST_Dataset training dataset
4. For the CIFAR10 dataset, the following additional keys need to be included:
   1. last_epoch_imgs: The training data augmented in the last epoch
   2. last_epoch_labels: The training data labels augmented in the last epoch
5. For Mini ImageNet dataset, the following additional keys need to be included:
   1. last_train_aug_data: The training data augmented in the last epoch
   2. last_train_aug_label: The training data labels augmented in the last epoch

### Parameter Explanation
In general, the main parameters that need to be modified are as follows:

#### Prune the model:
There are 3 programs for pruning models trained on MNIST or CIFAR-10:
1. pruning_filter_layers.py - For pruning models trained on MNIST
2. pruning_filter_layers_CIFAR10.py - For pruning models trained on CIFAR10  
3. pruning_filter_layers_resnet.py - For pruning ResNet models
##### The parameters used are the same and will be introduced below:
- case: Corresponds to the pre-written case in pretrained_models.py / pretrained_models_CIFAR10.py
- device: The device used
- sampling: The entropy estimation method used
- batch_size: Batch size (passing through the model / computing entropy)

#### Test the effect after pruning:
There are 3 programs for testing models trained on MNIST or CIFAR-10 after pruning:
1. pruning_filter_test_layers.py - For testing models trained on MNIST
2. pruning_filter_test_layers_CIFAR10.py - For testing models trained on CIFAR10  
3. pruning_filter_test_layers_resnet.py - For testing pruned ResNet
##### The parameters used are the same and will be introduced below:
- case: Corresponds to the pre-written case in pretrained_models.py / pretrained_models_CIFAR10.py
- device: The device used
- sampling: The entropy estimation method used
- batch_size: Batch size (passing through the model)
- IXZ_batch: The batch size used for computing Renyi entropy
- sigma_method: The method used for selecting the kernel width when estimating Renyi entropy


#### Finetune the pruned models
There are 5 programs for finetuning:
1. pruning_filter_finetune_layer_ver2.py: finetune the model with the pruning order. Only one layer will be pruned. 
2. pruning_filter_finetune_layer_specific.py: finetune the model with the pruning order. All layer can be pruned with given left nodes for each layer.
3. pruning_filter_finetune_layer_fixRatio.py: finetune the model with the pruning order. All layer can be pruned with given fixed pruning ratio for all layer
4. pruning_filter_finetune_layer_fixRatio_resnet.py: finetune the ResNet with the pruning order. All layer can be pruned with given fixed pruning ratio for all layer
5. pruning_filter_finetune_layer_only.py: finetune the pruned model.
