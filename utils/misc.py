import pickle


def save_obj(obj, data_dir, filename):
    with open(data_dir + filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(data_dir, filename):
    with open(data_dir + filename + '.pkl', 'rb') as f:
        return pickle.load(f)
