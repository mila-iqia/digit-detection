import numpy as np
import torch

from utils.boxes import extract_outer_box


class FirstCrop(object):
    """Crop the image such that all bounding boxes +30% in x,y are
       contained in the image.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, pad_size):

        self.pad_size = pad_size

    def __call__(self, sample):

        image = sample['image']
        labels = sample['metadata']['labels']
        boxes = sample['metadata']['boxes']

        outer_box = extract_outer_box(sample, padding=self.pad_size)
        outer_box = np.round(outer_box).astype('int')

        x1_tot, x2_tot, y1_tot, y2_tot = outer_box

        boxes_cropped = boxes
        boxes_cropped[:, 0:2] -= x1_tot
        boxes_cropped[:, 2:] -= y1_tot

        img_cropped = image.crop((x1_tot, y1_tot, x2_tot, y2_tot))

        metadata = {'boxes': boxes_cropped, 'labels': labels}

        return {'image': img_cropped, 'metadata': metadata}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        boxes = sample['metadata']['boxes']
        labels = sample['metadata']['labels']

        h, w = np.asarray(image).shape[:2]

        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image_scaled = image.resize((new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        boxes = boxes.astype('float64')
        boxes[:, :2] *= (new_w / w)
        boxes[:, 2:] *= (new_h / h)
        boxes = boxes.astype('int64')

        metadata = {'boxes': boxes, 'labels': labels}

        return {'image': image_scaled, 'metadata': metadata}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        image = sample['image']
        labels = sample['metadata']['labels']
        boxes = sample['metadata']['boxes']

        h, w = np.asarray(image).shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image_cropped = image.crop((left, top, left+new_w, top+new_h))

        boxes[:, 0:2] -= left
        boxes[:, 2:] -= top

        boxes[:, :2] = np.clip(boxes[:, :2], 0, new_w-1)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, new_h-1)

        metadata = {'boxes': boxes, 'labels': labels}

        return {'image': image_cropped, 'metadata': metadata}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image = sample['image']
        labels = sample['metadata']['labels']

        image = np.asarray(image)
        image = image - np.mean(image)
        assert image.shape == (54, 54, 3)

        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        # TODO
        # Process boxes

        labels = np.asarray(labels)

        # target is a 1x6 vector, where [0] is the number of digits and
        # targets[1:targets[0]] is the digit sequence.
        # i.e. the sequence 157 is represented by target [3,1,5,5,7,-1,-1]
        target = -np.ones(6)

        # First element of target is the number of digits in the image
        # A Target of 6 represents >5 digits in the image, i.e. we have
        # 7 classes. Refer to paper for more details.
        target[0] = min(len(labels), 6)

        # Here we identify at most 5 digits
        for jj in range(min(len(labels), 5)):

            target[jj+1] = labels[jj]

        target = torch.from_numpy(target).int()
#         metadata = {'boxes': boxes, 'labels': labels}

        return {'image': image,
                'target': target}
