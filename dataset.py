import os

import cv2
from torchvision import datasets
import torchvision.transforms as transforms

import paths
import albumentations as A
from albumentations.pytorch import ToTensorV2


def loader(p):
    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_dataset(dataset_name, path=paths.data_path):
    if dataset_name in ['amazon', 'webcam']:  # OFFICE-31
        train_transform = A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(45),
                A.OneOf([A.ColorJitter(),
                         A.RandomContrast(),
                         A.Blur(), A.CLAHE(), A.RandomGamma(), A.ChannelShuffle()]),
                A.CoarseDropout(),

                A.Normalize(),
                ToTensorV2()
            ]
        )
        test_transform = A.Compose(
            [
                A.Resize(256, 256),
                A.CenterCrop(224, 224),

                A.Normalize(),
                ToTensorV2(),
            ]
        )
        source_train = datasets.ImageFolder(os.path.join(path, dataset_name, f'source_train'),
                                            lambda x: train_transform(image=x)['image'], loader=loader)
        target_train = datasets.ImageFolder(os.path.join(path, dataset_name, f'target_train'),
                                            lambda x: train_transform(image=x)['image'], loader=loader)
        test = datasets.ImageFolder(os.path.join(path, dataset_name, f'test'), lambda x: test_transform(image=x)['image'],
                                    loader=loader)



    else:
        raise ValueError('Dataset %s not found!' % dataset_name)
    return source_train, target_train, test
