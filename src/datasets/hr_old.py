import pandas as pd
import numpy as np
import torch

from datetime import datetime
from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import TensorDataset

class HROLD_Dataset(TorchvisionDataset):

    def __init__(self, root:str, normal_class):

        super().__init__(root)
        self.normal_class = normal_class

        # get path to one patient
        csv_hr = "/workspace/HR_WearablesData/" + "P110465" + "/Orig_Fitbit_HR.csv"
        csv_st = "/workspace/HR_WearablesData/" + "P110465" + "/Orig_Fitbit_ST.csv"

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

        # convert string to datetime timestamp
        df_merged['datetime'] = pd.to_datetime(df_merged['datetime'], format='%Y-%m-%d %H:%M:%S')


        # make time column to convert datetime to interger
        df_merged['time'] = (df_merged['datetime'] - df_merged['datetime'][0])
        df_merged['time'] = df_merged['time'].apply(pd.to_timedelta)
        df_merged['time'] = (df_merged['time'].astype(int)) / 1000000000

        # make deltatime column to find interger time difference between current and previous point
        df_merged['deltatime'] = df_merged["time"].diff(1)

        # fill first value of deltatime with zero (maybe it should be avg of something else? idk)
        df_merged['deltatime'] = df_merged['deltatime'].fillna(0)

        # can drop columns of df_merged but just copied for easy access to prev results
        # shape: 1406831 * 4
        df_selected = df_merged.iloc[:, 0:4]
        df_selected['deltatime'] = df_merged['deltatime']

        # define variables to cut data into segments
        max_length = df_selected.shape[0]
        length = 11932
        overlap = 70
        start = length - overlap
        end = start + length

        # make 3d array for training
        x_final_train = np.array([np.array(df_selected.iloc[0:length, :].to_numpy()), np.array(df_selected.iloc[start:end, :].to_numpy())])
        while 1:
            start = start + length - overlap
            end = start + length
            if end - start == length and end < max_length:
                x_final_train = np.append(x_final_train, [np.array(df_selected.iloc[start:end, :].to_numpy())], axis=0)
            else:
                x_final_train = np.append(x_final_train, [np.array(df_selected.iloc[(max_length - length):(max_length+1), :].to_numpy())], axis=0)
                break  

        # load labels from file
        label_filepath = '/workspace/datasets/Health New Labeling.xlsx'
        df_label = pd.read_excel(label_filepath, sheet_name=0, engine="openpyxl")

        # patient_id
        patient_id = df_label.iloc[:, 0]
        # COVID-19 positive date
        label_positive = df_label.iloc[:, 1]
        label_positive = pd.to_datetime(label_positive, format='%Y-%m-%d')
        # symoptom date
        label_symptom = df_label.iloc[:, 2]
        for i in range(0, label_symptom.shape[0]):
            if label_symptom[i] == "-":
                label_symptom[i] = "1900-01-01"
        label_symptom = pd.to_datetime(label_symptom, format='%Y-%m-%d')

        # create y label array
        y_final_train = np.zeros(x_final_train.shape[0]) # label 0 is normal

        for i in range(0, x_final_train.shape[0]):
            for j in range(0, length):
                if x_final_train[i][j][0] == label_positive[0]:
                    y_final_train[i] = 1 # label 1 is anomaly
                    break

        index_final_train = np.arange(x_final_train.shape[0])

        train_set = TensorDataset(x_final_train, torch.Tensor(y_final_train), torch.Tensor(index_final_train))
        self.train_set = train_set


        # TODO first testing with train set to check if this dataloader works
        # test_set = TensorDataset(torch.Tensor(x_final_test), torch.Tensor(y_final_test), torch.Tensor(index_test))
        test_set = TensorDataset(x_final_train, torch.Tensor(y_final_train), torch.Tensor(index_final_train))
        self.test_set = test_set
