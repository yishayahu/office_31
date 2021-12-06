import os
import random
import shutil

import numpy as np

import paths


def split(part_ratio):
    path1 = paths.data_path
    img_len = []
    for domain in os.listdir(path1):
        domain = os.path.join(path1, domain)
        if os.path.exists(os.path.join(domain, 'train')):
            shutil.rmtree(os.path.join(domain, 'train'))
            shutil.rmtree(os.path.join(domain, 'val'))
            shutil.rmtree(os.path.join(domain, 'test'))
        os.makedirs(os.path.join(domain, 'train'))
        os.makedirs(os.path.join(domain, 'val'))
        os.makedirs(os.path.join(domain, 'test'))
        for class1 in os.listdir(os.path.join(domain, 'images')):

            os.makedirs(os.path.join(domain, 'train', class1))
            os.makedirs(os.path.join(domain, 'val', class1))
            os.makedirs(os.path.join(domain, 'test', class1))
            images = list(os.scandir(os.path.join(domain, 'images', class1)))
            img_len.append(len(images))

            random.shuffle(images)
            part_len = max(int(len(images) * part_ratio), 1)
            for img in images[:part_len]:
                shutil.copy(img.path, os.path.join(domain, 'train', class1, img.name))
            for img in images[part_len:2 * part_len]:
                shutil.copy(img.path, os.path.join(domain, 'val', class1, img.name))
            for img in images[2 * part_len:]:
                shutil.copy(img.path, os.path.join(domain, 'test', class1, img.name))
