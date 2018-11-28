import os

from PIL import Image
from torch.utils import data

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

        metadata = {'labels': labels, 'boxes': boxes}

        sample = {'image': image, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample
