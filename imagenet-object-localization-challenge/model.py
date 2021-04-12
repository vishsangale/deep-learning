import torch.nn as nn
import torch.nn.functional as F

from params import DatasetParams


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(DatasetParams.nr_channels, 16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 3))
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(3, 3))
        self.conv8 = nn.Conv2d(1024, 2048, kernel_size=(3, 3))

        self.fc1 = nn.Linear(2048, 1500)
        self.final = nn.Linear(1500, 1000)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        x = F.relu(self.fc1(x))
        x = F.softmax(self.final(x))
        return x
