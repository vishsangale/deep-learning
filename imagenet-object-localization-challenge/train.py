import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import imagenet

from model import VGGNet
data_root_dir = "/Users/vishsangale/workspace/imagenet-object-localization-challenge"

train_set = imagenet.ImageNet(data_root_dir, split="train")
train_loader = DataLoader(train_set)

valid_set = imagenet.ImageNet(data_root_dir, split="val")
valid_loader = DataLoader(valid_set)

m = VGGNet()

loss = nn.CrossEntropyLoss()


for train_batch, labels in train_loader:
    print(train_batch)