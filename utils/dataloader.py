import os

from PIL import Image
from torch.utils import data

from collections import OrderedDict
import numpy as np

from torchvision import transforms
from torch.utils.data import sampler, DataLoader

from utils.transforms import (
        CenterCrop, FirstCrop, Normalize, Rescale, RandomCrop, ToTensor)
from utils.misc import load_obj
from utils.boxes import extract_labels_boxes


class SVHNDataset(data.Dataset):

    def __init__(
                self, data_dir, metadata_filename,
                train=True, transform=None):
        '''
        SVHN Dataset.

        Parameters
        ----------
        data_dir : str
            Directory with all the images.
        metadata_filename: str
            Path to the metadata_filename.
        train: bool
            If true, use the train set otherwise use the test set.
        transform : callable, optional
            Optional transform to be applied on a sample.

        '''

        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        metadata = load_obj(metadata_filename)
        metadata = OrderedDict(metadata)

        self.metadata = metadata

    def __len__(self):
        '''
        Evaluate the length of the dataset object.

        Returns
        -------
        int
            The length of the dataset.

        '''
        return len(self.metadata)

    def __getitem__(self, index):
        '''
        Get an indexed item from the dataset and renerates one sample of data.

        Parameters
        ----------
        index : int
            The index of the dataset

        Returns
        -------
        sample: dict
            sample['image'] contains the image array.
            The type may be a torch.tensor or ndarray depending on transforms
            sample['metadata'] will contain the metadata associated to the
            image. It can be one of ['labels','boxes','filename']


        '''

        img_name = os.path.join(self.data_dir, self.metadata[index]['split'],
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


class ChunkSampler(sampler.Sampler):

    def __init__(self, num_samples, start=0, shuffle=False):
        '''
        Samples elements sequentially from some offset and eventually
        shuffle them.

        Parameters
        ----------
        num_samples: int
            # of desired datapoints
        start: int
            Offset where we should start selecting from
        shuffle: bool.
            If True, shuffle the samples.

        '''
        self.num_samples = num_samples
        self.start = start
        self.indices = range(self.start, self.start + self.num_samples)
        self.shuffle = shuffle

    def __iter__(self):
        '''
        Iterate over the offset.

        Returns
        -------
        int
            An indice.

        '''
        if self.shuffle:
            return iter(np.random.permutation(self.indices))
        else:
            return iter(self.indices)

    def __len__(self):
        '''
        Evaluate the length of the offset.

        Returns
        -------
        int
            The length of the offset.

        '''
        return len(self.indices)


def find_mean_std_per_channel(
            input_dir, metadata_filename,
            valid_split, transform, sample_size):
    '''
    Find the mean and std per channel of training images for normalization.
    '''
    train_dataset = SVHNDataset(data_dir=input_dir,
                                metadata_filename=metadata_filename,
                                train=True,
                                transform=transform)

    if sample_size != -1:
        train_num_samples = int((1 - valid_split) * sample_size)
    else:
        train_num_samples = int((1 - valid_split) * len(train_dataset))

    train_loader = DataLoader(train_dataset,
                              sampler=ChunkSampler(
                                num_samples=train_num_samples,
                                start=0,
                                shuffle=False),
                              batch_size=1,
                              shuffle=False,
                              num_workers=4)

    img_mean = []
    img_std = []
    for i, batch in enumerate(train_loader):
        inputs, _ = batch['image'], batch['target']
        x = inputs.data.cpu().numpy()[0]
        img_mean.append(x.reshape((x.shape[0], -1)).mean(axis=1))
        img_std.append(x.reshape((x.shape[0], -1)).std(axis=1))

    images_mean = np.array(img_mean).mean(0)
    images_std = np.array(img_std).std(0)
    print('Images mean: {}'.format(tuple(images_mean)))
    print('Images std: {}'.format(tuple(images_std)))
    return images_mean, images_std


def prepare_dataloaders(input_dir, metadata_filename, batch_size,
                        valid_split=0.2,
                        sample_size=-1,
                        train=True):
    '''
    Prepare the dataloader.

    Parameters
    ----------
    input_dir : str
        Directory with all the images.
    metadata_filename: str
            Path to the metadata_filename.
    batch_size : int
        Mini-batch size.
    valid_split : float
        Returns a validation split of %size; valid_split*100,
        valid_split should be in range [0,1].
        Default 0.2.
    sample_size : int
        Number of elements to use as sample size,
        for debugging purposes only. If -1, use all samples.
        Default -1.
    train: bool
        If true, use the train set otherwise use the test set.
        Default True.

    Returns
    -------
    if dataset_split in ['train']:
        train_loader: torch.utils.DataLoader
            Dataloader containing training data.
        valid_loader: torch.utils.DataLoader
            Dataloader containing validation data.

    if dataset_split in ['test']:
        test_loader: torch.utils.DataLoader
            Dataloader containing test data.

    '''

    # Define transformations
    firstcrop = FirstCrop(0.3)
    rescale = Rescale((64, 64))
    random_crop = RandomCrop((54, 54))
    center_crop = CenterCrop((54, 54))
    to_tensor = ToTensor()

    # Set basic transform
    train_transform = [firstcrop, rescale, random_crop, to_tensor]
    test_transform = [firstcrop, rescale, center_crop, to_tensor]

    # Find mean and std per channel for normalization
    if train:
        images_mean, images_std = find_mean_std_per_channel(
            input_dir,
            metadata_filename,
            valid_split,
            transforms.Compose(test_transform), sample_size)

    else:
        # Obtained from training set, avoids having to load it unnecessarily
        images_mean = (109.7994, 110.00522, 114.33739)
        images_std = (12.675092, 12.741672, 11.369844)
    # Define normalization
    normalize = Normalize(tuple(images_mean), tuple(images_std))

    # Data augmentation and normalization for training
    # Just normalization for test and validation
    train_transform.append(normalize)
    test_transform.append(normalize)
    data_transforms = {
        'train': transforms.Compose(
            train_transform),
        'test': transforms.Compose(
            test_transform)
        }

    if train:
        # Train dataset
        train_dataset = SVHNDataset(data_dir=input_dir,
                                    metadata_filename=metadata_filename,
                                    train=True,
                                    transform=data_transforms['train'])
        # Validation dataset
        valid_dataset = SVHNDataset(data_dir=input_dir,
                                    metadata_filename=metadata_filename,
                                    train=True,
                                    transform=data_transforms['test'])

        if sample_size != -1:
            train_num_samples = int((1 - valid_split) * sample_size)
            valid_num_samples = sample_size - train_num_samples
        else:
            train_num_samples = int((1 - valid_split) * len(train_dataset))
            valid_num_samples = len(train_dataset) - train_num_samples

        print('# of train examples: {}'.format(train_num_samples))
        print('# of valid examples: {}'.format(valid_num_samples))

        # Train dataset loader
        train_loader = DataLoader(
            train_dataset,
            sampler=ChunkSampler(
                num_samples=train_num_samples,
                start=0,
                shuffle=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4)
        # Validation dataset loader
        valid_loader = DataLoader(
            valid_dataset,
            sampler=ChunkSampler(
                num_samples=valid_num_samples,
                start=train_num_samples,
                shuffle=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4)

        return train_loader, valid_loader

    else:
        # Test dataset
        test_dataset = SVHNDataset(data_dir=input_dir,
                                   metadata_filename=metadata_filename,
                                   train=False,
                                   transform=data_transforms['test'])

        if sample_size != -1:
            test_num_samples = sample_size
        else:
            test_num_samples = len(test_dataset)

        print('# of test examples: {}'.format(test_num_samples))

        # Test dataset loader
        test_loader = DataLoader(
            test_dataset,
            sampler=ChunkSampler(
                num_samples=test_num_samples,
                start=0,
                shuffle=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4)

        return test_loader
