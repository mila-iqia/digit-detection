# Data

Data for the projet will be provided directly by the Mila team. It is based
off of the original data found in [SVHN](http://ufldl.stanford.edu/housenumbers/).

Briefly, the Street View House Numbers (SVHN) Dataset is an open-source dataset
containing around 200k street numbers. The dataset annotation contains bounding
boxes and class labels for individual digits, giving about 600k digits total.

There are limitations in these data in regards to our practical case:
- It's zoom on the numbers and consequently there is a lack of
background.
- There is no negative examples (i.e. no images without numbers).

However, it's a good dataset to start with.

Note: We ask for the project that you only use the data provided by Mila as we
will keep the test set blind. Do not pick up the official test set available
online. We reserve the right to reject a model if we have valid reasons to
believe that the test set was used to build/refine the model.

## Download the data
On Helios a shared disk contain the `train.tar.gz` file. We have formatted the
original bounding box data so that it can easily be imported in python through
pickle files.

You should extract them within the `data/SVHN` directory.

To extract the data you can run:
- `mkdir -p $HOME/digit-detection/data/SVHN`
- `cp '/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train_metadata.pkl' $HOME'/digit-detection/data/SVHN/'`
- `cp '/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train.tar.gz' $HOME'/digit-detection/data/SVHN/'`
- `tar -xzf $HOME'/digit-detection/data/SVHN/train.tar.gz' -C $HOME'/digit-detection/data/SVHN/'`

When you run your experiment you might want to use local storage to speed up
your experiments. If the storage you want to use is shared with other users
don't forget to create a directory under your username to put your files.
See calcul quebec [using avaiable storage](https://wiki.calculquebec.ca/w/Utiliser_l%27espace_de_stockage/en?setlang=fr)
for more informations.

Once extracted, the data directory structure should look like the following:

```
digit_detection/
├── data
│   ├── README.md
│   ├── SVHN
│   │   ├── train
│   │   │    ├── 1.png
│   │   │    ├── 2.png
│   │   │    ├── 3.png
│   │   │    └── ...
│   │   └── train_metadata.pkl
```
The metatada pickle files contain a python dictionary with all metadata and
labels associated to each image. Here is a sample script to load the metadata
in to a python dict.

```
with open('train_metadata.pkl', 'rb') as f:
    train_metadata = pickle.load(f)
```

Each key in `train_metadata` will contain a filename and all associated
metadata. The filename is with respect to the directory it's in, and metadata
contains four 5 fields:
* `label` - which lists all digits present in image, in order
* `height`,`width`,`top`,`left` which list the corresponding pixel information
about the digits bounding boxes.

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
