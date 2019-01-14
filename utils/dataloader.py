import os
from pathlib import Path


from PIL import Image
from torch.utils import data

import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.transforms import FirstCrop, Rescale, RandomCrop, ToTensor
from utils.utils import load_obj
from utils.boxes import extract_labels_boxes


class SVHNDataset(data.Dataset):

    def __init__(self, metadata, data_dir, transform=None):
        """
        Args:
            labels (dict): Dictionary containing all labels and metadata
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = metadata
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        '''
        Parameters
        ----------
        index : int
            The index of the dataset

        Returns
        -------
        X : PIL objet

        y : dict
            The metadata associated to the image in dict form.

        '''
        'Generates one sample of data'

        img_name = os.path.join(self.data_dir,
                                self.metadata[index]['filename'])

        # Load data and get raw metadata (labels & boxes)
        image = Image.open(img_name)
        metadata_raw = self.metadata[index]['metadata']

        labels, boxes = extract_labels_boxes(metadata_raw)

        metadata = {'labels': labels, 'boxes': boxes, 'filename': img_name}

        sample = {'image': image, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample


def prepare_dataloaders(dataset_split,
                        batch_size=32,
                        datadir=None,
                        sample_size=None,
                        valid_split=0.8):
    '''
    dataset_split (str) : Any of 'train', 'extra', 'test'

    valid_split (float) : Returns a validation split of %size
    valid_split*100, should be in range [0,1]

    sample_size (int) : Number of elements to use as sample size,
    for debugging purposes only.

    '''

    assert dataset_split in ['train', 'test', 'extra'], "check dataset_split"

    datadir = Path(datadir)

    metadata_filename = datadir / (dataset_split + '_metadata.pkl')

    metadata = load_obj(metadata_filename)

    dataset_path = datadir / dataset_split

    firstcrop = FirstCrop(0.3)
    rescale = Rescale((64, 64))
    random_crop = RandomCrop((54, 54))
    to_tensor = ToTensor()

    # Declare transformations

    transform = transforms.Compose([firstcrop,
                                    rescale,
                                    random_crop,
                                    to_tensor])

    dataset = SVHNDataset(metadata,
                          data_dir=dataset_path,
                          transform=transform)

    indices = np.arange(len(metadata))
    indices = np.random.permutation(indices)

    # Only use a sample amount of data
    if sample_size:
        indices = indices[:sample_size]

    if dataset_split in ['train', 'extra']:

        train_idx = indices[:round(valid_split*len(indices))]
        valid_idx = indices[round(valid_split*len(indices)):]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        # Prepare a train and validation dataloader
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  sampler=train_sampler)

        valid_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  sampler=valid_sampler)

        return train_loader, valid_loader

    elif dataset_split in ['test']:

        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
        # Prepare a test dataloader
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=4,
                                 sampler=test_sampler)

        return test_loader
