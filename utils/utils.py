'''
Convert .mat file provided with SVHN dataset to python dict. Inspired by this
thread:
https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn

Step 1:  Download SVHN data in to data/SVHN/train
'''

import pickle
import h5py


def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
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
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def aggregate_data(index, hdf5_data):

    image_id = get_name(index, hdf5_data)
    labels = get_box_data(index, hdf5_data)

    # Convert label 10 to label 0 for digit 0
    if 10 in labels['label']:
        labels['label'] = [0 if x == 10 else x for x in labels['label']]

    metadata = {}

    metadata['filename'] = image_id
    metadata['metadata'] = labels

    return metadata


def save_obj(obj, data_dir, filename):
    with open(data_dir + filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(data_dir, filename):
    with open(data_dir + filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def convert_mat_file(data_dir, filename_mat, filename_out='labels'):
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
