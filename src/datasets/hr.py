import pandas as pd
import numpy as np
import torch
import os.path

from glob import glob
from datetime import datetime
from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import TensorDataset

class HR_Dataset(TorchvisionDataset):

    def __init__(self, root:str, normal_class):

        super().__init__(root)
        self.normal_class = normal_class

        # x_array = [[[0 for k in range(3)] for j in range(11932)]]

        # load lists of participant ids
        # id_fb, id_nfb = load_id('/workspace/HR_WearablesData/')
        # id_fb = np.load("/workspace/fitbit_id.npy")
        # id_nfb = np.load("/workspace/nonfitbit_id.npy")
        # id_anomalies = load_labels('/workspace/datasets/Health New Labeling.xlsx')

        # df = load_fitbit_data(id_fb[0])
        # x_array = cut_to_same_length(df, x_array)
        # y_array = np.zeros(x_array.shape[0])
        # index_array = np.arange(x_array.shape[0])

        print("start")
        dim1_train = pd.read_csv("/workspace/dim1_train.txt").to_numpy()
        dim2_train = pd.read_csv("/workspace/dim2_train.txt").to_numpy()
        dim3_train = pd.read_csv("/workspace/dim3_train.txt").to_numpy()

        dim1_test = pd.read_csv("/workspace/dim1_test.txt").to_numpy()
        dim2_test = pd.read_csv("/workspace/dim2_test.txt").to_numpy()
        dim3_test = pd.read_csv("/workspace/dim3_test.txt").to_numpy()

        labels_train = pd.read_csv("/workspace/labels_train.csv").to_numpy()
        labels_test = pd.read_csv("/workspace/labels_test.csv").to_numpy()
        print("all files loaded.")

        print("train set: ")
        print(dim1_train.shape)
        print(dim2_train.shape)
        print(dim3_train.shape)
        print(len(labels_train))

        print("test set: ")
        print(dim1_test.shape)
        print(dim2_test.shape)
        print(dim3_test.shape)
        print(len(labels_test))

        index_array_train = np.arange(len(labels_train))
        index_array_test = np.arange(len(labels_test))

        x_array_train = np.dstack([dim1_train, dim2_train, dim3_train])
        x_array_test = np.dstack([dim1_test, dim2_test, dim3_test])
        print("creating datasets...")

        train_set = TensorDataset(torch.Tensor(x_array_train), torch.Tensor(labels_train), torch.Tensor(index_array_train))
        self.train_set = train_set

        test_set = TensorDataset(torch.Tensor(x_array_test), torch.Tensor(labels_test), torch.Tensor(index_array_test))
        self.test_set = test_set
        print("done.")









def load_fitbit_data(fitbit_id):
    """
    input: participant id who used fitbit device

    """
    # get path to one patient
    csv_hr = "/workspace/HR_WearablesData/" + fitbit_id + "/Orig_Fitbit_HR.csv"
    csv_st = "/workspace/HR_WearablesData/" + fitbit_id + "/Orig_Fitbit_ST.csv"
 
    # load csv files as dataframes
    df_hr = pd.read_csv(csv_hr)
    df_st = pd.read_csv(csv_st)

    # merge dataframes so that it shows only data with a heart rate value.
    df_merged = pd.merge(df_hr, df_st, how='left', on='datetime')

    # make a new column to get heart rate over steps (hros)
    df_merged['hros'] = df_merged['steps']
    # fill in NaN value steps with 1 because should not be divided by zero
    df_merged['hros'] = df_merged['hros'].fillna(1)
    # divide heart rate by number of steps. if no step value, divided by one so it is original hr
    df_merged['hros'] = df_merged['heartrate'] / df_merged['hros']

    # fill in those NaN value steps with 0
    df_merged['steps'] = df_merged['steps'].fillna(0)

    return df_merged.loc[:, ['heartrate', 'steps', 'hros']]


def cut_to_same_length(df, x_array):
    max_length = df.shape[0]
    length = 11932
    overlap = 50
    start = length - overlap
    end = start + length
    x_array = np.append(x_array, [np.array(df.iloc[0:length, :].to_numpy()), np.array(df.iloc[start:end, :].to_numpy())], axis=0)
    while 1:
        start = start + length - overlap
        end = start + length
        if end - start == length and end < max_length:
            x_array = np.append(x_array, [np.array(df.iloc[start:end, :].to_numpy())], axis=0)
        else:
            x_array = np.append(x_array, [np.array(df.iloc[(max_length - length):(max_length+1), :].to_numpy())], axis=0)
            break

    return x_array          
        



def load_nonfitbit_data(nonfitbit_id):
    """
    """
    # for n in nonfitbit_id:
    return nonfitbit_id


def load_labels(label_filepath):
    """
    input: path to excel file with information about anomalies
    df_label.iloc[:, 0] is participant id
    df_label.iloc[:, 1] is COVID-19 test date
    df_label.iloc[:, 2] is symptom date
    df_label.iloc[:, 3] is retrospective or prospective
    df_label.iloc[:, 4] is devices used by participants
    84 rows in total = 84 participants tested positive = anomalies
    output: series of ids with anomalies
    """
    df_label = pd.read_excel(label_filepath, sheet_name=0, engine="openpyxl")
    
    return df_label.iloc[:, 0]



######################################################################

def load_id(path):
    """
    input: path to folders with folder names as patient ids.
    output: 
    """
    all_directories = glob(path + "*")
    fitbit_id = []
    nonfitbit_id = []
    for d in all_directories:
        if os.path.isfile(d + "/Orig_NonFitbit_HR.csv"):
            df = pd.read_csv(d + "/Orig_NonFitbit_HR.csv")
            if (df.shape[0] >= 11932):
                nonfitbit_id.append(d[28:-1])
        else:
            df = pd.read_csv(d + "/Orig_Fitbit_HR.csv")
            if (df.shape[0] >= 11932):
                fitbit_id.append(d[28:-1])
        
    return fitbit_id, nonfitbit_id