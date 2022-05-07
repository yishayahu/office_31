import csv
import os

import cv2
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import datasetops as do
from pathlib import Path
from office31.download import download_and_extract_office31
from torch.utils.data import Dataset

import paths
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
import argparse


class Chexpert(Dataset):
    def __init__(self,mode):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        image_list_file = Path('/mnt/dsi_vol1/shaya_data/chexpertV2/')
        self.dataroot = Path('/mnt/dsi_vol1/shaya_data/chexpertV2/')
        if mode == 'train':
            self.imgs = pickle.load(open(image_list_file/'train_images.p','rb'))
            self.labels = pickle.load(open(image_list_file/'train_labels.p','rb'))
        else:
            self.imgs = pickle.load(open(image_list_file/'test_images.p','rb'))[:4000]
            self.labels = pickle.load(open(image_list_file/'test_labels.p','rb'))[:4000]
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        if mode =='train':
            transformList.append(transforms.RandomResizedCrop((256,256)))
            transformList.append(transforms.RandomHorizontalFlip())
        else:
            transformList.append(transforms.Resize((256, 256)))

        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""

        image_name = self.imgs[index]
        image_name = self.dataroot / image_name
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label),False

    def __len__(self):
        return len(self.imgs)






class CXR14(Dataset):
    """CXR14 dataset"""

    def __init__(self, mode='train'):

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        if mode =='train':
            transformList.append(transforms.RandomResizedCrop((256,256)))
            transformList.append(transforms.RandomHorizontalFlip())
        else:
            transformList.append(transforms.Resize((256, 256)))

        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)
        image_list_file = Path('/mnt/dsi_vol1/shaya_data/CX14/')
        self.dataroot = image_list_file /'images'
        if mode == 'train':
            self.imgs = pickle.load(open(image_list_file/'train_images.p','rb'))
            self.labels = pickle.load(open(image_list_file/'train_labels.p','rb'))
        else:
            self.imgs = pickle.load(open(image_list_file/'test_images.p','rb'))[:4000]
            self.labels = pickle.load(open(image_list_file/'test_labels.p','rb'))[:4000]


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img_path = self.dataroot / img_path
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, torch.FloatTensor(self.labels[idx]),True







def get_dataset(dataset_name,target_size, seed=1):
    test_split_seed = 42  # hard-coded
    target_size = int(target_size)
    if dataset_name == 'chexpert':
        source_train = Chexpert(mode='train')
        target_train = Chexpert(mode='train')
        target_train.imgs = target_train.imgs[:target_size]
        test = Chexpert(mode='test')
    else:
        assert dataset_name == 'cxr14'
        source_train = CXR14(mode='train')
        target_train = CXR14(mode='train')
        target_train.imgs = target_train.imgs[:target_size]
        test = CXR14(mode='test')


    return source_train, target_train, test
