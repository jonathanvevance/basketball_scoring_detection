import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class verySimpleNet(nn.Module):
    def __init__(self, train_loader):
        super(verySimpleNet, self).__init__()

        self.resnet_fe = models.resnet18(pretrained = True)
        self.resnet_fe = nn.Sequential(*list(self.resnet_fe.children())[:-1])
        for param in self.resnet_fe.parameters():
            param.requires_grad = False

        sample_image_batch, __ = iter(train_loader).next()
        linear_dim = self.get_linear_dim(sample_image_batch)
        self.fc = nn.Linear(linear_dim, 1)

    def get_linear_dim(self, x):

        if len(x.shape) == 5:
            x = x[0] # 5D -> 4D (videos) + batch_dim

        x = self.resnet_fe(x)
        x = torch.flatten(x, 1)
        return x.shape[1]

    def forward(self, x):
        x = self.resnet_fe(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class simpleConvNet(nn.Module):
    def __init__(self, train_loader, use_fe = True):
        super(simpleConvNet, self).__init__()
        self.use_fe = use_fe

        # feature extractor
        if self.use_fe:
            self.resnet_fe = models.resnet18(pretrained = True)
            self.resnet_fe = nn.Sequential(*list(self.resnet_fe.children())[:-3])
            for param in self.resnet_fe.parameters():
                param.requires_grad = False

        # conv layers
        if self.use_fe:
            self.conv1 = nn.Conv2d(256, 32, 3, 1)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, 1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)

        sample_image_batch, __ = iter(train_loader).next()
        linear_dim = self.get_linear_dim(sample_image_batch)
        self.fc1 = nn.Linear(linear_dim, 1)

    def get_linear_dim(self, x):

        if len(x.shape) == 5:
            x = x[0] # 5D -> 4D (videos) + batch_dim

        x = self.resnet_fe(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x.shape[1]

    def forward(self, x):

        x = self.resnet_fe(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x