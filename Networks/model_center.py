import torchvision.models as models_imagenet
import torch
from Networks.resnet18_imagenet import resnet18, resnet18_Tanh

def get_model(args, torchvision = False):
    if torchvision:
        if args.pretrained and args.arch == "resnet18":
            model =  models_imagenet.resnet18(weights=models_imagenet.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Linear(model.fc.in_features, 100)
            print("pretrained model")
            return model
        
        if args.datasets in ['ImageNet'] :
            return models_imagenet.__dict__[args.arch]()
        elif args.datasets in ['mini_imageNet']:
            model = models_imagenet.__dict__[args.arch]()
            model.fc = torch.nn.Linear(model.fc.in_features, 100)
            return model
    else:
        assert args.arch == "resnet18", ">>> Apart from ResNet18, none of them have been implemented."

        if args.activate.lower() == "relu":
            model_fuct = resnet18
        elif args.activate.lower() == "tanh":
            model_fuct = resnet18_Tanh
        else:
            print(">>>> this activate function is not implemented!")

        model = model_fuct(pretrained=args.pretrained)

        if args.datasets in ['mini_imageNet']:
            model.fc = torch.nn.Linear(model.fc.in_features, 100)
            
        return model

                

    
