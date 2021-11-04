import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base.base_net import BaseNet

class RS_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(30, 32, 3, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256, 64, bias=False)
    
    def forward(self, x):
        print("0: ", x.size()) # [138, 206, 3]
        x = x.unsqueeze(3)
        print("1: ", x.size()) # [138, 206, 3, 1]
        x = self.conv1(x)
        print("2: ", x.size()) # [138, 32, 5, 3]
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        print("3: ", x.size()) # [138, 32, 2, 1]
        x = self.conv2(x)
        print("4: ", x.size()) # [138, 64, 4, 3]
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        print("5: ", x.size()) # [138, 64, 2, 1]
        x = self.conv3(x)
        print("6: ", x.size()) # [138, 128, 4, 3]
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        print("7: ", x.size()) # [138, 128, 2, 1]
        x = x.view(x.size(0), -1)
        print("8: ", x.size()) # [138, 256]
        x = self.fc1(x)
        print("9: ", x.size()) # [138, 64]
        return x

class RS_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(30, 32, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256, 64, bias=False)
        self.bn1d = nn.BatchNorm1d(64, eps=1e-04, affine=False)

        self.deconv1 = nn.ConvTranspose2d(int(64 / (2 * 2)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 30, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        print("00: ", x.size()) # [39, 30, 6]
        x = x.unsqueeze(3)
        print("01: ", x.size()) # [39, 30, 6, 1]
        x = self.conv1(x)
        print("02: ", x.size()) # [39, 32, 8, 3]
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        print("03: ", x.size()) # [39, 32, 4, 1]
        x = self.conv2(x)
        print("04: ", x.size()) # [39, 64, 6, 3]
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        print("05: ", x.size()) # [39, 64, 3, 1]
        x = self.conv3(x)
        print("06: ", x.size()) # [39, 128, 5, 3]
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        print("07: ", x.size()) # [39, 128, 2, 1]
        x = x.view(x.size(0), -1)
        print("08: ", x.size()) # [39, 256]
        x = self.bn1d(self.fc1(x))
        print("09: ", x.size()) # [39, 64]
        x = x.view(x.size(0), int(64 / (2 * 2)), 2, 2)
        print("10: ", x.size()) # [39, 16, 2, 2]
        x = F.leaky_relu(x)
        print("11: ", x.size()) # [39, 16, 2, 2]
        x = self.deconv1(x)
        print("12: ", x.size()) # [39, 128, 2, 2]
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        print("13: ", x.size()) # [39, 128, 4, 4]
        x = self.deconv2(x)
        print("14: ", x.size()) # [39, 64, 2, 2]
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        print("15: ", x.size()) # [39, 64, 4, 4]
        x = self.deconv3(x)
        print("16: ", x.size()) # [39, 32, 2, 2]
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        print("17: ", x.size()) # [39, 32, 4, 4]
        x = self.deconv4(x)
        print("18: ", x.size()) # [39, 30, 2, 2]
        x = torch.sigmoid(x)
        print("19: ", x.size()) # [39, 30, 2, 2]
        return x[:,:,:,0]