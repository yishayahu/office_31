import os

import cv2
from torchvision import datasets
import torchvision.transforms as transforms
import datasetops as do
from pathlib import Path
from office31.download import download_and_extract_office31
from torch.utils.data import Dataset

import paths
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OfficeDs(Dataset):
    def __init__(self, transform, inner_ds):
        self.inner_ds = inner_ds
        self.transform = transform
        self.name_to_num = {'tape_dispenser': 0, 'bike_helmet': 1, 'paper_notebook': 2, 'stapler': 3, 'calculator': 4, 'printer': 5, 'back_pack': 6, 'desk_chair': 7, 'desktop_computer': 8, 'laptop_computer': 9, 'bike': 10, 'bookcase': 11, 'phone': 12, 'punchers': 13, 'pen': 14, 'projector': 15, 'ring_binder': 16, 'ruler': 17, 'headphones': 18, 'letter_tray': 19, 'bottle': 20, 'scissors': 21, 'desk_lamp': 22, 'mouse': 23, 'trash_can': 24, 'monitor': 25, 'speaker': 26, 'file_cabinet': 27, 'keyboard': 28, 'mug': 29, 'mobile_phone': 30}

    def __len__(self):
        return len(self.inner_ds)

    @staticmethod
    def loader(p):
        image = cv2.imread(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index):
        img_path, target = self.inner_ds[index]
        img = self.loader(img_path)
        img = self.transform(image=img)['image']
        return img, self.name_to_num[target]


def get_dataset(dataset_name,source_size,target_size, office_path=paths.data_path, seed=1):
    test_split_seed = 42  # hard-coded
    source_size = int(source_size)
    target_size = int(target_size)
    num_source_per_class = source_size if dataset_name == "amazon" else int(source_size * 0.4)
    num_target_per_class = target_size
    office_path = Path(office_path)
    if not office_path.exists():
        download_and_extract_office31(office_path)
        assert office_path.exists()

    source = do.from_folder_class_data(office_path / dataset_name / "images").named(
        "s_data", "s_label"
    )
    target = do.from_folder_class_data(office_path / dataset_name / "images").named(
        "t_data", "t_label"
    )
    source_train = source.shuffle(seed)
    if source_size != -1:

        source_train = source.filter(
            s_label=do.allow_unique(num_source_per_class)
        )
    else:
        print('using all')

    target_test, target_trainval = target.split(
        fractions=[0.3, 0.7], seed=test_split_seed  # hard-coded seed
    )
    target_train, target_val = target_trainval.shuffle(seed).split_filter(
        t_label=do.allow_unique(num_target_per_class)
    )

    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(),
            A.OneOf([A.ColorJitter(),
                     A.RandomBrightnessContrast(),
                     A.Blur(), A.CLAHE(), A.RandomGamma(), A.ChannelShuffle(), A.HueSaturationValue(), A.RGBShift(),
                     A.FancyPCA()]),
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
    source_train = OfficeDs(transform=train_transform, inner_ds=source_train)
    target_train = OfficeDs(transform=train_transform, inner_ds=target_train)
    test = OfficeDs(transform=test_transform, inner_ds=target_test)

    return source_train, target_train, test
