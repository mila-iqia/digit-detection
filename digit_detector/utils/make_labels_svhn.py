'''
Script that helps make labels from the coco format to the SVHN format used
in bloc 1 + 2 of the course
'''
import json
from utils.misc import load_obj
from utils.misc import save_obj

with open('data/Avenue/Humanware_v1_1553272293/train/instances_train.json') as json_file:
    coco_data_train = json.load(json_file)

with open('data/Avenue/Humanware_v1_1553272293/valid/instances_valid.json') as json_file:
    coco_data_val = json.load(json_file)

with open('data/Avenue/Humanware_v1_1553272293/test/instances_test.json') as json_file:
    coco_data_test = json.load(json_file)

with open('data/Avenue/Humanware_v1_1553272293/labels.json') as json_file:
    original_data = json.load(json_file)

train_svhn = load_obj('data/SVHN/train_metadata_split.pkl')

train_avenue = {}
valid_avenue = {}
test_avenue = {}

use_split = False

split = 'train'

idx = 0
count = 0
for idx in range(len(coco_data_train['images'])):
    content = {}
    filename = coco_data_train['images'][idx]['file_name']
    content['filename'] = filename

    if use_split:
        content['split'] = split

    house_number = str(int(original_data[count]['house_number'][0]))
    bbox = coco_data_train['annotations'][idx]['bbox']
    x, y, w, h = bbox

    top = []
    left = []
    height = []
    width = []
    label = []

    x = round(x)
    y = round(y)
    w = round(w)
    h = round(h)

    metadata = {}

    if len(house_number) == 1:
        print(len(house_number))

    for jj in range(len(house_number)):
        left.append(x)
        top.append(y)
        height.append(h)
        width.append(w)
        label.append(int(house_number[jj]))

    metadata['top'] = top
    metadata['left'] = left
    metadata['width'] = width
    metadata['height'] = height
    metadata['label'] = label

    content['metadata'] = metadata
    train_avenue[idx] = content

    count += 1


split = 'valid'
for idx in range(len(coco_data_val['images'])):
    content = {}
    filename = coco_data_val['images'][idx]['file_name']
    content['filename'] = filename

    if use_split:
        content['split'] = split


    house_number = str(int(original_data[count]['house_number'][0]))
    bbox = coco_data_val['annotations'][idx]['bbox']
    x, y, w, h = bbox

    x = round(x)
    y = round(y)
    w = round(w)
    h = round(h)

    metadata = {}
    
    top = []
    left = []
    height = []
    width = []
    label = []

    for jj in range(len(house_number)):
        left.append(x)
        top.append(y)
        height.append(h)
        width.append(w)
        label.append(int(house_number[jj]))

    metadata['top'] = top
    metadata['left'] = left
    metadata['width'] = width
    metadata['height'] = height
    metadata['label'] = label

    content['metadata'] = metadata
    valid_avenue[idx] = content

    count += 1



split = 'test'
for idx in range(len(coco_data_test['images'])):
    content = {}
    filename = coco_data_test['images'][idx]['file_name']
    content['filename'] = filename

    if use_split:
        content['split'] = split


    house_number = str(int(original_data[count]['house_number'][0]))
    bbox = coco_data_test['annotations'][idx]['bbox']
    x, y, w, h = bbox

    x = round(x)
    y = round(y)
    w = round(w)
    h = round(h)

    metadata = {}
    
    top = []
    left = []
    height = []
    width = []
    label = []

    for jj in range(len(house_number)):
        left.append(x)
        top.append(y)
        height.append(h)
        width.append(w)
        label.append(int(house_number[jj]))

    metadata['top'] = top
    metadata['left'] = left
    metadata['width'] = width
    metadata['height'] = height
    metadata['label'] = label

    content['metadata'] = metadata
    test_avenue[idx] = content

    count += 1


if use_split:
    save_obj(train_avenue, 'data/avenue_train_metadata_split.pkl')
    save_obj(valid_avenue, 'data/avenue_val_metadata_split.pkl')
    save_obj(test_avenue, 'data/avenue_test_metadata_split.pkl')

else:
    save_obj(train_avenue, 'data/avenue_train_metadata.pkl')
    save_obj(valid_avenue, 'data/avenue_val_metadata.pkl')
    save_obj(test_avenue, 'data/avenue_test_metadata.pkl')
