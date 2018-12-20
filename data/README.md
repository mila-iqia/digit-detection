# Data

Data for the projet will be provided directly by the Mila team. It is based off of the original data found in [SVHN](http://ufldl.stanford.edu/housenumbers/). We ask for the project that you only use the data provided by Mila as we will keep the test set blind. We have formatted the original bounding box data so that it can easily be imported in python through pickle files.

A shared disk will contain a `train.tar.gz` and `extra.tar.gz` files. You should extract them within the `data/SVHN` directory. Once extracted, the data directory structure should look like the following:

```
digit_detection/
├── data
│   ├── README.md
│   ├── SVHN
│   │   ├── extra
│   │   │    ├── 1.png
│   │   │    ├── 1.png
│   │   │    ├── 3.png
│   │   │    └── ...
│   │   ├── extra_metadata.pkl
│   │   ├── train
│   │   │    ├── 1.png
│   │   │    ├── 2.png
│   │   │    ├── 3.png
│   │   │    └── ...
│   │   └── train_metadata.pkl
```
The metatada pickle files contain a python dictionary with all metadata and labels associated to each image. Here is a sample script to load the metadata in to a python dict. 

```
with open('train_metadata.pkl', 'rb') as f:
    train_metadata = pickle.load(f)
```

Each key in `train_metadata` will contain a filename and all associated metadata. The filename is with respect to the directory it's in, and metadata contains four 5 fields: 
* `label` - which lists all digits present in image, in order
* `height`,`width`,`top`,`left` which correspond to pixel information about the bounding boxes.


```
{0: {'filename': '1.png',
  'metadata': {'height': [219, 219],
   'label': [1, 9],
   'left': [246, 323],
   'top': [77, 81],
   'width': [81, 96]}},
 1: {'filename': '2.png',
  'metadata': {'height': [32, 32],
   'label': [2, 3],
   'left': [77, 98],
   'top': [29, 25],
   'width': [23, 26]}},
...
```
