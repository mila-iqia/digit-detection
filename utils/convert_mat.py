from utils import convert_mat_file

if __name__ == '__main__':

    split = 'extra'
    data_dir = '../data/SVHN/' + split + '/'
    filename_mat = '../data/SVHN/' + split + '/digitStruct.mat'

    convert_mat_file(data_dir, filename_mat)
