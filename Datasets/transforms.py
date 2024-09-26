# mocdify from the official code of:
# @inproceedings{fifty2023context,
#   title={Context-Aware Meta-Learning},
#   author={Fifty, Christopher and Duan, Dennis and Junkins, Ronald Guenther and Amid, Ehsan and Leskovec, Jure and Re, Christopher and Thrun, Sebastian},
#   booktitle={The Twelfth International Conference on Learning Representations},
#   year={2023}
# }

import random
import sys
import timm

from PIL import Image, ImageFilter
from torchvision import transforms

# from pyprojroot import here as project_root

# sys.path.insert(0, str(project_root()))


def get_timm_transform(timm_model):
    data_config = timm.data.resolve_model_data_config(timm_model)
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    test_transform = timm.data.create_transform(**data_config, is_training=False)
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_dino_transform():
    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ],
                               p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # first global crop
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.14, 1.), interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        GaussianBlur(1.0),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.14, 1.), interpolation=Image.BICUBIC),
        normalize,
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_resnet_transform():
    transform = transforms.Compose([
        transforms.RandomChoice(
            [transforms.Resize(256),
             transforms.Resize(480)]),
        transforms.RandomCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return {
        'train_transform': transform,
        'test_transform': transform,
    }


def get_empty_transform():
    return {
        'train_transform': transforms.Compose([]),
        'test_transform': transforms.Compose([]),
    }

# the augenment of Few-Shot Incremental Learning with Continually Evolved Classifiers:
def get_CEC_transform():
    image_size = 84
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])


    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


class GaussianBlur(object):
    """
  Apply Gaussian Blur to the PIL image.
  """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)))
