import os
import random
import shutil

import numpy as np

import paths


def split(part_ratio):
    part_ratio_name = int(str(part_ratio).split('.')[1])
    path1 = paths.data_path
    img_len = []
    for domain in os.listdir(path1):
        if domain == 'outs':
            continue
        domain = os.path.join(path1, domain)
        os.makedirs(os.path.join(domain, f'train_{part_ratio_name}'))
        os.makedirs(os.path.join(domain, f'val_{part_ratio_name}'))
        os.makedirs(os.path.join(domain, f'test_{part_ratio_name}'))
        for class1 in os.listdir(os.path.join(domain, 'images')):

            os.makedirs(os.path.join(domain, f'train_{part_ratio_name}', class1))
            os.makedirs(os.path.join(domain, f'val_{part_ratio_name}', class1))
            os.makedirs(os.path.join(domain, f'test_{part_ratio_name}', class1))
            images = list(os.scandir(os.path.join(domain, 'images', class1)))
            img_len.append(len(images))

            random.shuffle(images)
            part_len = max(int(len(images) * part_ratio), 1)
            for img in images[:part_len]:
                shutil.copy(img.path, os.path.join(domain, f'train_{part_ratio_name}', class1, img.name))
            for img in images[part_len:2 * part_len]:
                shutil.copy(img.path, os.path.join(domain, f'val_{part_ratio_name}', class1, img.name))
            for img in images[2 * part_len:]:
                shutil.copy(img.path, os.path.join(domain, f'test_{part_ratio_name}', class1, img.name))
if __name__ == '__main__':

    split(0.05)
    split(0.1)
    split(0.2)