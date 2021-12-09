import os
import random
import shutil

import numpy as np

import paths


def split():

    path1 = paths.data_path
    img_len = []
    for domain in os.listdir(path1):
        if domain == 'webcam':
            part_ratio = 0.34
            part_ratio_test = 1/8 + 0.01
        elif domain == 'amazon':
            part_ratio = 2/9 +0.01
            part_ratio_test = 1/30 +0.01
        if domain == 'outs' or domain == 'dslr':
            continue
        domain = os.path.join(path1, domain)
        if os.path.exists(os.path.join(domain, f'source_train')):
            shutil.rmtree(os.path.join(domain, f'source_train'))
            shutil.rmtree(os.path.join(domain, f'target_train'))
            shutil.rmtree(os.path.join(domain, f'test'))
        os.makedirs(os.path.join(domain, f'source_train'))
        os.makedirs(os.path.join(domain, f'target_train'))
        os.makedirs(os.path.join(domain, f'test'))

        for class1 in os.listdir(os.path.join(domain, 'images')):

            os.makedirs(os.path.join(domain, f'source_train', class1))
            os.makedirs(os.path.join(domain, f'target_train', class1))
            os.makedirs(os.path.join(domain, f'test', class1))
            images = list(os.scandir(os.path.join(domain, 'images', class1)))
            img_len.append(len(images))

            random.shuffle(images)
            source_len = max(int(len(images) * part_ratio), 1)
            target_len = max(int(len(images) * part_ratio_test), 1)
            for img in images[:source_len]:
                shutil.copy(img.path, os.path.join(domain, f'source_train', class1, img.name))
            for img in images[source_len: source_len+ target_len]:
                shutil.copy(img.path, os.path.join(domain, f'target_train', class1, img.name))
            for img in images[source_len+ target_len:]:
                shutil.copy(img.path, os.path.join(domain, f'test', class1, img.name))
if __name__ == '__main__':
    split()