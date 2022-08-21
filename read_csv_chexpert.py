import csv
import pickle
import random

import numpy as np

train_images = []
test_images = []
test_labels = []
train_labels = []
all_labels = []
all_images = []
def get_csv(p1):
    with open(p1, "r") as f:
        csvReader = csv.reader(f)
        next(csvReader, None)
        k=0
        lines = list(csvReader)
        random.shuffle(lines)
        for line in lines:
            k+=1
            image_name= line[0]
            label = line[5:]

            for i in range(14):
                if label[i]:
                    a = float(label[i])
                    if a == 1:
                        label[i] = 1
                    elif a == -1:
                        label[i] = 1

                    else:
                        label[i] = 0
                else:
                    label[i] = 0
            if random.random() <0.1:
                test_images.append(image_name.replace('CheXpert-v1.0-small/',''))
                test_labels.append(label)
            else:
                train_images.append(image_name.replace('CheXpert-v1.0-small/',''))
                train_labels.append(label)
get_csv('/mnt/dsi_vol1/shaya_data/chexpertV2/train.csv')
get_csv('/mnt/dsi_vol1/shaya_data/chexpertV2/valid.csv')
print(len(train_images))
print(len(train_labels))
print(len(test_images))
print(len(test_labels))

pickle.dump(train_images,open('/mnt/dsi_vol1/shaya_data/chexpertV2/train_images.p','wb'))
pickle.dump(train_labels,open('/mnt/dsi_vol1/shaya_data/chexpertV2/train_labels.p','wb'))
pickle.dump(test_labels,open('/mnt/dsi_vol1/shaya_data/chexpertV2/test_labels.p','wb'))
pickle.dump(test_images,open('/mnt/dsi_vol1/shaya_data/chexpertV2/test_images.p','wb'))