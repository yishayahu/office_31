import os
import random
import shutil

import numpy as np

random.seed(10)
path1 = r'C:\Users\Y\PycharmProjects\domain_adaptation_images'
img_len = []
for domain in os.listdir(path1):
    domain = os.path.join(path1,domain)
    os.makedirs(os.path.join(domain, 'train'),exist_ok=True)
    os.makedirs(os.path.join(domain, 'val'),exist_ok=True)
    os.makedirs(os.path.join(domain, 'test'),exist_ok=True)
    for class1 in os.listdir(os.path.join(domain,'images')):

        os.makedirs(os.path.join(domain, 'train',class1),exist_ok=True)
        os.makedirs(os.path.join(domain, 'val',class1),exist_ok=True)
        os.makedirs(os.path.join(domain, 'test',class1),exist_ok=True)
        images = list(os.scandir(os.path.join(domain,'images',class1)))
        img_len.append(len(images))

        random.shuffle(images)
        part_len = len(images)//5
        for img in images[:part_len]:
            shutil.copy(img.path,os.path.join(domain,'train',class1,img.name))
        for img in images[part_len:2*part_len]:
            shutil.copy(img.path,os.path.join(domain,'val',class1,img.name))
        for img in images[2*part_len:]:
            shutil.copy(img.path,os.path.join(domain,'test',class1,img.name))
