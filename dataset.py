import os

from torchvision import datasets
import torchvision.transforms as transforms

import paths


def get_dataset(dataset_name, path=paths.data_path):
    if dataset_name in ['amazon', 'webcam']:  # OFFICE-31
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.ColorJitter(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        source_train = datasets.ImageFolder(os.path.join(path, dataset_name, f'source_train'), data_transforms['train'])
        target_train = datasets.ImageFolder(os.path.join(path, dataset_name, f'target_train'), data_transforms['train'])
        test = datasets.ImageFolder(os.path.join(path, dataset_name, f'test'), data_transforms['test'])



    else:
        raise ValueError('Dataset %s not found!' % dataset_name)
    return source_train, target_train, test
