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
lbls = {'Cardiomegaly': 0, 'Emphysema': 1, 'Effusion': 2, 'Hernia': 3, 'Infiltration': 4
    , 'Mass': 5, 'Nodule': 6, 'Atelectasis': 7, 'Pneumothorax': 8, 'Pleural_Thickening': 9
    , 'Pneumonia': 10, 'Fibrosis': 11, 'Edema': 12, 'Consolidation': 13}


kk = 0
with open('/mnt/dsi_vol1/shaya_data/CX14/Data_Entry_2017_v2020.csv', "r") as f:
    csvReader = csv.reader(f)
    x=4
    next(csvReader, None)
    k=0
    lines = list(csvReader)
    random.shuffle(lines)
    for line in lines:
        k+=1
        image_name= line[0]
        labels = line[1].split('|')
        ll = np.zeros(14)
        for l in labels:
            if l == 'No Finding':
                continue
            ll[lbls[l]] = 1
        if random.random() < 0.1:
            test_images.append(image_name)
            test_labels.append(ll)
        else:
            train_images.append(image_name)
            train_labels.append(ll)
print(len(train_images))
print(len(train_labels))
print(len(test_images))
print(len(test_labels))

pickle.dump(train_images,open('/mnt/dsi_vol1/shaya_data/CX14/train_images.p','wb'))
pickle.dump(train_labels,open('/mnt/dsi_vol1/shaya_data/CX14/train_labels.p','wb'))
pickle.dump(test_labels,open('/mnt/dsi_vol1/shaya_data/CX14/test_labels.p','wb'))
pickle.dump(test_images,open('/mnt/dsi_vol1/shaya_data/CX14/test_images.p','wb'))