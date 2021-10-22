"""
implementation sources:
https://shashikachamod4u.medium.com/excel-csv-to-pytorch-dataset-def496b6bcc1
and
https://github.com/sniezek/keras-character-trajectories-classification
"""

import pandas as pd
import torch
import copy
import random
import numpy as np
from torch.utils.data import Dataset
# from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from base.torchvision_dataset import TorchvisionDataset
# from torch.utils.data import Subset
# from PIL import Image
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms

zero_point_that_can_be_skipped = '0,0,0'
single_sequence_end = ',,'
padding_vector = [0.0, 0.0, 0.0]
longest_sequence_length_with_trimmed_zeros = 182
longest_sequence_length = 205
shortest_sequence_length = 109
# output
number_of_character_classes = 20  # a b c d e g h l m n o p q r s u v w y z

class CT_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):

        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 20))
        self.outlier_classes.remove(normal_class)
        
        x = get_input_data()
        y = get_output_data()

        test_count = int(0.1 * len(x))

        x_y = list(zip(x, y))
        random.shuffle(x_y)
        x, y = zip(*x_y)

        # train_set = MyCT(x_values=x, y_values=y, idx=test_count, train=True)
        # test_set = MyCT(x_values=x, y_values=y, idx=test_count, train=False)

        train_set = Dataset(np.array(x[test_count:]), np.array(y[test_count:]))
        test_set = Dataset(np.array(x[:test_count]), np.array(y[:test_count])        
        
        print("checkpoint")
        print(train_set)

        # in this order: x_train, y_train, x_test, y_test
        # np.array(x[test_count:]), np.array(y[test_count:]), np.array(x[:test_count]), np.array(y[:test_count])

        # self.X_train = torch.tensor(x_train, dtype=torch.float32)
        # self.y_train = torch.tensor(y_train)

        # train_idx_normal = get_target_label_idx(train_set.train_labels, self.normal_classes)
        # self.train_set = Subset(train_set, train_idx_normal)

        # self.test_set = MyCT(root=self.root)
    
def get_input_data():
    x = []
    with open('../data/input.csv') as f:
        single_sequence = []
        for point in f:
            if zero_point_that_can_be_skipped in point:
                continue

            if single_sequence_end in point:
                for i in range(longest_sequence_length_with_trimmed_zeros - len(single_sequence)):
                    single_sequence.insert(0, padding_vector)

                x.append(copy.deepcopy(single_sequence))
                single_sequence = []
                continue

            single_sequence.append([])
            for point_element in point.split(','):
                single_sequence[-1].append(float(point_element))
    return x


def get_output_data():
    y = []
    cnt = 0
    with open('../data/output.txt') as f:
        for character_class in f.readlines()[0].split('|'):
            y.append(int(character_class) - 1)
            cnt += 1
    return np_utils.to_categorical(y, number_of_character_classes)


class MyCT(Dataset):

    def __init__(self, x_values, y_values, idx, train=True):
        self.x_values = x_values
        self.y_values = y_values
        self.idx = idx
        self.train = train

    def __getitem__(self, idx):
        if self.train:
            X = x_values[idx:]
            y = y_values[idx:]
        else:
            X = x_values[:idx]
            y = y_values[:idx]
        
        return X, y

