import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MNIST_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        print("0: ", x.size()) # [200, 1, 28, 28]
        x = self.conv1(x)
        print("1: ", x.size()) # [200, 8, 28, 28]
        x = self.pool(F.leaky_relu(self.bn1(x)))
        print("2: ", x.size()) # [200, 8, 14, 14]
        x = self.conv2(x)
        print("3: ", x.size()) # [200, 4, 14, 14]
        x = self.pool(F.leaky_relu(self.bn2(x)))
        print("4: ", x.size()) # [200, 4, 7, 7]
        x = x.view(x.size(0), -1)
        print("5: ", x.size()) # [200, 196]
        x = self.fc1(x)
        print("6: ", x.size()) # [200, 32]
        return x


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        print("000: ", x.size()) # [200, 1, 28, 28]
        x = self.conv1(x)
        print("001: ", x.size()) # [200, 8, 28, 28]
        x = self.pool(F.leaky_relu(self.bn1(x)))
        print("002: ", x.size()) # [200, 8, 14, 14]
        x = self.conv2(x)
        print("003: ", x.size()) # [200, 4, 14, 14]
        x = self.pool(F.leaky_relu(self.bn2(x)))
        print("004: ", x.size()) # [200, 4, 7, 7]
        x = x.view(x.size(0), -1)
        print("005: ", x.size()) # [200, 196]
        x = self.fc1(x)
        print("006: ", x.size()) # [200, 32]
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        print("007: ", x.size()) # [200, 2, 4, 4]
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        print("008: ", x.size()) # [200, 2, 8, 8]
        x = self.deconv1(x)
        print("009: ", x.size()) # [200, 4, 8, 8]
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        print("010: ", x.size()) # [200, 4, 16, 16]
        x = self.deconv2(x)
        print("011: ", x.size()) # [200, 8, 14, 14]
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        print("012: ", x.size()) # [200, 8, 28, 28]
        x = self.deconv3(x)
        print("013: ", x.size()) # [200, 1, 28, 28]
        x = torch.sigmoid(x)
        print("014: ", x.size()) # [200, 1, 28, 28]

        return x
