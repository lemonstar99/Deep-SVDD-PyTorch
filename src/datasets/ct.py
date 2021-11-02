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
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset
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

    def __init__(self, root: str, normal_class=2):

        super().__init__(root)
        # self.root = root
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 20))
        self.outlier_classes.remove(normal_class)
        
        x = get_input_data()
        y = get_output_data()
        y_new = []
        
        for i in range(0,2858):
            if y[i][normal_class] == 1:
                y_new.append(0)
            else:
                y_new.append(1)

        test_count = int(0.1 * len(x))

        x_y = list(zip(x, y))
        random.shuffle(x_y)
        x, y = zip(*x_y)

        """

        # y_train_new = get_target_label_idx(np.array(y[test_count:]), self.normal_classes)
        # y_test_new = get_target_label_idx(np.array(y[:test_count]), self.normal_classes)
    
        x_train_transformed = global_contrast_normalization(torch.Tensor(np.array(x[test_count:])))
        x_test_transformed = global_contrast_normalization(torch.Tensor(np.array(x[:test_count])))

        # print(x_test_transformed.type())
        # print(y_transformed.type())

        train_set = TensorDataset(x_train_transformed, torch.Tensor(np.array(y[test_count:])), torch.Tensor(np.arange(test_count, 2858)))
        test_set = TensorDataset(x_test_transformed, torch.Tensor(np.array(y[:test_count])), torch.Tensor(np.arange(0, test_count)))   

        self.train_set = train_set
        self.test_set = test_set
        
        """

        train_set = TensorDataset(torch.Tensor(np.array(x[test_count:])), torch.Tensor(np.array(y_new[test_count:])), torch.Tensor(np.arange(285, 2858)))
        test_set = TensorDataset(torch.Tensor(np.array(x[:test_count])), torch.Tensor(np.array(y_new[:test_count])), torch.Tensor(np.arange(0, 285)))

        self.train_set = train_set
        self.test_set = test_set   

        """
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1'))])
        
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCT(root=self.root, idx=test_count, x_values=x, y_values=y, train=True,
                            transform=transform, target_transform=target_transform)
        
        train_idx_normal = get_target_label_idx(y, self.normal_classes)
        self.train_set = TensorDataset(train_set, train_idx_normal)
            
        self.test_set = MyCT(root=self.root, idx=test_count, x_values=x, y_values=y, train=False,
                                transform=transform, target_transform=target_transform)
        """

        # in this order: x_train, y_train, x_test, y_test
        # np.array(x[test_count:]), np.array(y[test_count:]), np.array(x[:test_count]), np.array(y[:test_count])

    
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
    # copy from mnist
    def __init__(self, root, idx, x_values, y_values, train, transform, target_transform):
        self.root = root
        self.idx = idx
        self.x_values = x_values
        self.y_values = y_values
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.train is True:
            # x, y = self.train_data[index], self.train_labels[index]
            x = x_values[idx:]
            y = y_values[idx:]
        else:
            # x, y = self.test_data[index], self.test_labels[index]
            x = x_values[:idx]
            y = y_values[:idx]
        
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y, index
    """
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
        
        return X, y, idx
    """

