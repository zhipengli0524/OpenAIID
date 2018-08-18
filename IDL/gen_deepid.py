from deepid_generate import DeepIDGenerator
import pickle
import gzip
import os
from layers import relu

def load_data_xy(data_path):
    """load data from data file
    Args:
        data_path    path of file
    Returns:
        x            [img_vector, ...]
        y            [img_label, ...]
    Raises:
        None
    """
    fin = open(data_path, 'rb')
    x, y = pickle.load(fin)
    fin.close()
    return x, y


def get_params_from_file(params_file):
    """get params from file
    Args:
        paramsms_file   path of file
    Returns:
        list          [params...]
    Raises:
        None
    """
    if os.path.exists(params_file):
        f = gzip.open(params_file)
        dumped_params = pickle.load(f)
        f.close()
        return dumped_params
    return []


if __name__ == "__main__":
    nkerns = [20, 40, 60, 80]
    n_hidden = 160

    data_path = "./0.pkl"
    x, y = load_data_xy(data_path)

    params_file = "./model_params/params.bin"
    exist_params = get_params_from_file(params_file)
    if len(exist_params) != 0:
        exist_params = exist_params[-1]
    else:
        print 'no params in param_file'
        exit(-1)

    deepid = DeepIDGenerator(exist_params)
    deepid.layer_params(nkerns, 1)
    deepid.build_layer_architecture(n_hidden, relu)

    cnt = 0
    for index in range(10):
        vec = x[index]
        lab = y[index]
        did = deepid.generate_deepid([vec])
        print "label: %d" % lab
        print "deepid: "
        print did





    


