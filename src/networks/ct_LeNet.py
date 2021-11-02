import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base.base_net import BaseNet

# import data


class CT_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        # DID find the rep_dim of CT
        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        # TODO make it similar to NeuTraL AD
        # find the number of conv layers in CT
        # 1d convolution layers with ReLU activation
        # no batch normalization is applied
        # input size [1,3,182]

        # DID no batch normalization.
        self.conv1 = nn.Conv2d(182, 32, 3, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256, 200, bias=False)

        # self.relu = nn.ReLU()
        # self.conv1 = nn.Conv1d(182, 32, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv2 = nn.Conv1d(32, 64, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv4 = nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv5 = nn.Conv1d(256, 256, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv6 = nn.Conv1d(256, 256, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv6.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.fc1 = nn.Conv1d(256, 64, kernel_size=1, stride=1, bias=False)
  
    # TODO forward layers will be same as above
    def forward(self, x):
        # x = np.expand_dims(x, 1)
        print("0: ", x.size())
        x = x.unsqueeze(3)
        print("1: ", x.size())
        x = self.conv1(x)
        print("2: ", x.size())
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        print("3: ", x.size())
        x = self.conv2(x)
        print("4: ", x.size())
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        print("5: ", x.size())
        x = self.conv3(x)
        print("6: ", x.size())
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        print("7: ", x.size())
        # x = x.view(x.size(0), -1)
        # print("8: ", x.size())
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = self.fc1(x)
        print("9: ", x.size())
        return x


class CT_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        # DID "rep_dim" same as above
        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        # DID works without batch normalization
        self.conv1 = nn.Conv2d(182, 32, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256, 200, bias=False)
        self.bn1d = nn.BatchNorm1d(200, eps=1e-04, affine=False)

        # self.relu = nn.ReLU()
        # self.conv1 = nn.Conv1d(182, 32, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv2 = nn.Conv1d(32, 64, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv4 = nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv5 = nn.Conv1d(256, 256, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.conv6 = nn.Conv1d(256, 256, kernel_size=1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.conv6.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.fc1 = nn.Conv1d(256, 64, kernel_size=1, stride=1, bias=False)
        # self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # TODO Decoder
        self.deconv1 = nn.ConvTranspose2d(int(200/ (2 * 2)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 182, 4, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

        # self.deconv1 = nn.ConvTranspose1d(64, 256, 1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.deconv2 = nn.ConvTranspose1d(256, 256, 1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.deconv3 = nn.ConvTranspose1d(256, 128, 1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.deconv4 = nn.ConvTranspose1d(128, 64, 1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.deconv5 = nn.ConvTranspose1d(64, 32, 1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.deconv6 = nn.ConvTranspose1d(32, 182, 1, stride=2, bias=False)
        # nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))

    # TODO
    def forward(self, x):
        # DID add conv layers. remove unsqueeze bc dimension matches now
        # print("e", x.size()) # [200, 182, 3]
        x = x.unsqueeze(3)
        # print("e", x.size()) # [200, 182, 3, 1]
        x = self.conv1(x)
        # print("e", x.size()) # [200, 32, 5, 3]
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        # print("e", x.size()) # [200, 32, 2, 1]
        x = self.conv2(x)
        # print("e", x.size()) # [200, 64, 4, 3]
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        # print("e", x.size()) # [200, 64, 2, 1]
        x = self.conv3(x)
        # print("e", x.size()) # [200, 128, 4, 3]
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        # print("e", x.size()) # [200, 128, 2, 1]
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        """
        x = self.bn1d(self.fc1(x))
        x = F.leaky_relu(x)
        """
        x = x.view(x.size(0), -1)
        # print("a", x.size()) # [200, 256]
        x = self.bn1d(self.fc1(x))
        # print("a", x.size()) # [200, 200]
        x = x.view(x.size(0), int(200/ (2 * 2)), 2, 2)
        # print("a", x.size()) # [200, 50, 2, 2]
        x = F.leaky_relu(x)
        # print("a", x.size()) # [200, 50, 2, 2]
       
        x = self.deconv1(x)
        # print("d", x.size()) # [200, 128, 2, 2]
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        # print("d", x.size()) # [200, 128, 4, 4]
        x = self.deconv2(x)
        # print("d", x.size()) # [200, 64, 2, 2]
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        # print("d", x.size()) # [200, 64, 4, 4]
        x = self.deconv3(x)
        # print("d", x.size()) # [200, 32, 2, 2]
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        # print("d", x.size()) # [200, 32, 4, 4]
        x = self.deconv4(x)
        # print("d", x.size()) # [200, 128, 3, 3]
        # x = self.deconv5(x)
        # x = self.deconv6(x)
        x = torch.sigmoid(x)
        # print("final: ", x.size()) # [200, 128, 3, 3]
        return x[:,:,:,0]
