'''
Convert .mat file provided with SVHN dataset to python dict. Inspired by this
thread:
https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
Step 1:  Download SVHN data in to data/SVHN/train
'''

import h5py
from utils.misc import save_obj


def get_box_data(index, hdf5_data):
    '''
    Get `left, top, width, height` of each picture.

    Parameters
    ----------
    index : int
        Index.
    hdf5_data : ndarray
        ndarray representing the hdf5 file.

    Returns
    -------
    meta_data : dict
        Dictionary with the meta data corresponding to the index. The
        dictionary contains four 5 fields:
        * `label` - which lists all digits present in image, in order
        * `height`,`width`,`top`,`left` which list the corresponding
        pixel information about the digits bounding boxes.

    '''
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        '''
        Description.

        Parameters
        ----------
        name : str
            Key in the dict.
        obj : obj
            Object.

        '''
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def get_name(index, hdf5_data):
    '''
    Description.

    Parameters
    ----------
    index : int
        Index.
    hdf5_data : ndarray
        ndarray representing the hdf5 file.

    Returns
    -------
    str
        Filename corresponding to the index.

    '''
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def aggregate_data(index, hdf5_data):
    '''
    Description.

    Parameters
    ----------
    index : int
        Index.
    hdf5_data: ndarray
        ndarray representing the hdf5 file.

    Returns
    -------
    metadata : dict
        Each key in metadata will contain a filename and all associated
        metadata. The filename is with respect to the directory it's in,
        and metadata contains four 5 fields:
        * `label` - which lists all digits present in image, in order
        * `height`,`width`,`top`,`left` which list the corresponding pixel
        information about the digits bounding boxes.

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
        'width': [23, 26]}}, ...

    '''
    image_id = get_name(index, hdf5_data)
    labels = get_box_data(index, hdf5_data)

    # Convert label 10 to label 0 for digit 0
    if 10 in labels['label']:
        labels['label'] = [0 if x == 10 else x for x in labels['label']]

    metadata = {}

    metadata['filename'] = image_id
    metadata['metadata'] = labels

    return metadata


def convert_mat_file(data_dir, filename_mat, filename_out='labels'):
    '''
    Convert a .mat file into .pkl file.

    Parameters
    ----------
    data_dir : str
        Directory with all the images.
    filename_mat : str
        Absolute path to the metadata .mat file.
    filename_out : str
        Absolute path to the metadata pickle file.

    '''
    mat_data = h5py.File(filename_mat)
    dataset_size = mat_data['/digitStruct/name'].size

    # Save all metadata in a dict
    metadata = {}

    for index in range(dataset_size):

        metadata[index] = aggregate_data(index, mat_data)

        if index % 5000 == 0:
            print(index)

    print("Saving metadata dict ...")

    # Save to pickle file
    save_obj(metadata, data_dir, filename_out)


if __name__ == '__main__':

    split = 'extra'
    data_dir = '../data/SVHN/' + split + '/'
    filename_mat = '../data/SVHN/' + split + '/digitStruct.mat'

    convert_mat_file(data_dir, filename_mat)
