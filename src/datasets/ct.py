"""
from https://github.com/sniezek/keras-character-trajectories-classification
"""

import copy
import random

import numpy as np
from keras.utils import np_utils

class CT_Dataset(TorchvisionDataset):
    # input
    zero_point_that_can_be_skipped = '0,0,0'
    single_sequence_end = ',,'
    padding_vector = [0.0, 0.0, 0.0]
    longest_sequence_length_with_trimmed_zeros = 182
    longest_sequence_length = 205
    shortest_sequence_length = 109
    # output
    number_of_character_classes = 20  # a b c d e g h l m n o p q r s u v w y z

    def get_data(test_fraction):
        x = get_input_data()
        y = get_output_data()

        x_y = list(zip(x, y))
        random.shuffle(x_y)
        x, y = zip(*x_y)

        test_count = int(test_fraction * len(x))
        return np.array(x[test_count:]), np.array(y[test_count:]), np.array(x[:test_count]), np.array(y[:test_count])


    def get_input_data():
        x = []
        with open('/data/input.csv') as f:
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
        with open('/data/output.txt') as f:
            for character_class in f.readlines()[0].split('|'):
                y.append(int(character_class) - 1)

        return np_utils.to_categorical(y, number_of_character_classes)



"""
from torch.utils.data import Subset
from PIL import Image
# TODO CT is not in torchvision
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
        # min_max = [(-28.94083453598571, 13.802961825439636),
        #            (-6.681770233365245, 9.158067708230273),
        #            (-34.924463588638204, 14.419298165027628),
        #            (-10.599172931391799, 11.093187820377565),
        #            (-11.945022995801637, 10.628045447867583),
        #            (-9.691969487694928, 8.948326776180823),
        #            (-9.174940012342555, 13.847014686472365),
        #            (-6.876682005899029, 12.282371383343161),
        #            (-15.603507135507172, 15.2464923804279),
        #            (-6.132882973622672, 8.046098172351265)]
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
        
        train_set = MyCT(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCT(root=self.root, train=False, download=True,
                                  transform=transform, target_transform=target_transform)


class MyCT(CT):
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
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

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
