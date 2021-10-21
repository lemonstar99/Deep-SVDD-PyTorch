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
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import Subset
from PIL import Image
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms

class CT_Dataset(TorchvisionDataset):

    def __init__(self):
        zero_point_that_can_be_skipped = '0,0,0'
        single_sequence_end = ',,'
        padding_vector = [0.0, 0.0, 0.0]
        longest_sequence_length_with_trimmed_zeros = 182
        longest_sequence_length = 205
        shortest_sequence_length = 109
        # output
        number_of_character_classes = 20  # a b c d e g h l m n o p q r s u v w y z
        
        x = get_input_data()
        y = get_output_data()

        x_y = list(zip(x, y))
        random.shuffle(x_y)
        x, y = zip(*x_y)

        test_count = int(test_fraction * len(x))
        # in this order: x_train, y_train, x_test, y_test
        # np.array(x[test_count:]), np.array(y[test_count:]), np.array(x[:test_count]), np.array(y[:test_count])

        # self.X_train = torch.tensor(x_train, dtype=torch.float32)
        # self.y_train = torch.tensor(y_train)

        train_set = MyCT(root=self.root, train=True, download=True,
                              transform=None, target_transform=None)
        
        train_idx_normal = get_target_label_idx(train_set.train_labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCT(root=self.root, train=False, download=True,
                                  transform=None, target_transform=None)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
def get_input_data():
        x = []
        with open('data/input.csv') as f:
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
    with open('data/output.txt') as f:
        for character_class in f.readlines()[0].split('|'):
            y.append(int(character_class) - 1)

    return np_utils.to_categorical(y, number_of_character_classes)

class MyCT(CT_Dataset):
    # Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample.

    def __init__(self, *args, **kwargs):
        super(MyCT, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Override the original method of the CIFAR10 class.
        # Args:
        #     index (int): Index
        # Returns:
        #     triple: (image, target, index) where target is index of the target class.
        
        if self.train:
            img, target = self.np.array(x[test_count:]), self.np.array(y[test_count:])
        else:
            img, target = self.np.array(x[:test_count]), self.np.array(y[:test_count])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        # TODO our dataset does not need to be converted to an image. how should I modify this?

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed






"""
from https://github.com/sniezek/keras-character-trajectories-classification

    def get_data(test_fraction):
        x = get_input_data()
        y = get_output_data()

        

"""


"""
from torch.utils.data import Subset
from PIL import Image
# from torchvision.datasets import CIFAR10
# from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms

class CT_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        # DID what is normal_class in the input of function?
        # normal_class is the one class chosen from different classes to be the normal one
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 20))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        # min_max
        # TODO calculate min_max (CT has 20 classes so 20 rows with 2 columns)

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        # DID what does "*3" represent?
        # multiplication represents the number of dimensions of input data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # TODO train and test data is already separated in CT dataset

"""
