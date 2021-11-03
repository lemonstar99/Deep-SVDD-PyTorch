import pandas as pd
import numpy as np
import torch

from scipy.io import arff
from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import TensorDataset

class EP_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class):

        super().__init__(root)
        self.n_classes = 2
        self.normal_class = normal_class

        # train set
        #load data file path
        url1_train = '../data/epilepsy/EpilepsyDimension1_TRAIN.arff'
        url2_train = '../data/epilepsy/EpilepsyDimension2_TRAIN.arff'
        url3_train = '../data/epilepsy/EpilepsyDimension3_TRAIN.arff'

        # get x and y as dataframe
        x_dim1_train, target_train = get_data(url1_train)
        x_dim2_train, __ = get_data(url2_train)
        x_dim3_train, __ = get_data(url3_train)

        # combine 3 dimensions of x
        x_final_train = np.dstack([x_dim1_train, x_dim2_train, x_dim3_train])
        # process output y and produce index
        y_final_train, index_train = get_target(target_train)

        train_set = TensorDataset(torch.Tensor(x_final_train), torch.Tensor(y_final_train), torch.Tensor(index_train))
        self.train_set = train_set

        # set up testing set
        url1_test = '../data/epilepsy/EpilepsyDimension1_TEST.arff'
        url2_test = '../data/epilepsy/EpilepsyDimension2_TEST.arff'
        url3_test = '../data/epilepsy/EpilepsyDimension3_TEST.arff'

        x_dim1_test, target_test = get_data(url1_test)
        x_dim2_test, __ = get_data(url2_test)
        x_dim3_test, __ = get_data(url3_test)

        x_final_test = np.dstack([x_dim1_test, x_dim2_test, x_dim3_test])
        y_final_test, index_test = get_target(target_test)

        test_set = TensorDataset(torch.Tensor(x_final_test), torch.Tensor(y_final_test), torch.Tensor(index_test))
        self.test_set = test_set


def get_data(url):
    """
    input: path to arff data file
    This function loads the arff file, then converts into dataframe.
    The dataframe is then split into x and y.
    output: x is dataframe object without the last column. y is series.
    """
    loaded = arff.loadarff(url)
    df = pd.DataFrame(loaded[0])
    
    # dropping the last column of dataframe
    # it is still a dataframe object
    x = df.iloc[:, :-1].to_numpy()

    # getting last column as series, not dataframe object
    # as dataframe object is using iloc[:, -1:]
    y = df.iloc[:, -1]

    return x, y


def get_target(y, normal_class):
    """
    input: pandas series. last column of dataframe.
    This function converts the byte string of series and compare to each classification group
    Each class is represented as a number.
    output: returns numpy array of numbers and index array
    """
    y_new = []
    y_temp = []
    idx = []
    length = len(y)

    for i in range(0, length):
        if y[i].decode('UTF-8') == 'EPILEPSY':
            y_temp.append(0)
        elif y[i].decode('UTF-8') == 'SAWING':
            y_temp.append(1)
        elif y[i].decode('UTF-8') == 'RUNNING':
            y_temp.append(2)
        elif y[i].decode('UTF-8') == 'WALKING':
            y_temp.append(3)
        idx.append(i)

    for i in range(0, length):
        if y_temp[i] == normal_class:
            y_new.append(0) # normal
        else:
            y_new.append(1) # anomaly

    return np.array(y_new), np.array(idx)