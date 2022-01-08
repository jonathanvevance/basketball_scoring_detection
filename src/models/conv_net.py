"""Classes for neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class verySimpleNet(nn.Module):
    """
    Very simple neural network that takes a pretrained Resnet18's layers except the
    last and connects an FC layer with 1 output unit. Only the FC layer is trained.
    """
    def __init__(self, train_loader = None):
        """Init function."""
        super(verySimpleNet, self).__init__()

        self.resnet_fe = models.resnet18(pretrained = True)
        self.resnet_fe = nn.Sequential(*list(self.resnet_fe.children())[:-1])
        for param in self.resnet_fe.parameters():
            param.requires_grad = False

        if train_loader is not None:
            sample_batch = iter(train_loader).next()
            if isinstance(sample_batch, dict):
                sample_image_batch = sample_batch['X']
            else:
                sample_image_batch, __ = sample_batch
            linear_dim = self.get_linear_dim(sample_image_batch)
        else:
            linear_dim = 512

        self.fc = nn.Linear(linear_dim, 1)

    def get_linear_dim(self, x):
        """Get dimensions of flattened (after Conv) vector."""

        # Read docstrings in src/data/dataset.py
        # to understand why these inputs can be 5D.
        # Convolutional layers require 4D inputs.

        if len(x.shape) == 5: # if 5D input
            x = x[0]

        x = self.resnet_fe(x)
        x = torch.flatten(x, 1)
        return x.shape[1]

    def forward(self, x):
        """Run forward propagation."""
        x = self.resnet_fe(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class simpleConvNet(nn.Module):
    """
    Simple convolutional neural network that takes a pretrained Resnet18's layers except the
    last 3 and connects 2 convolutional layers and an FC layer with 1 output unit. The added
    convolutional layers and the FC layer are trained.
    """
    def __init__(self, train_loader = None):
        """Init function."""
        super(simpleConvNet, self).__init__()

        # feature extractor
        self.resnet_fe = models.resnet18(pretrained = True)
        self.resnet_fe = nn.Sequential(*list(self.resnet_fe.children())[:-3])
        for param in self.resnet_fe.parameters():
            param.requires_grad = False

        # conv layers
        self.conv1 = nn.Conv2d(256, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)

        if train_loader is not None:
            sample_batch = iter(train_loader).next()
            if isinstance(sample_batch, dict):
                sample_image_batch = sample_batch['X']
            else:
                sample_image_batch, __ = sample_batch
            linear_dim = self.get_linear_dim(sample_image_batch)
        else:
            linear_dim = 256 #

        self.fc1 = nn.Linear(linear_dim, 1)

    def get_linear_dim(self, x):
        """Get dimensions of flattened (after Conv) vector."""

        # Read docstrings in src/data/dataset.py
        # to understand why these inputs can be 5D.
        # Convolutional layers require 4D inputs.

        if len(x.shape) == 5: # if 5D input
            x = x[0]

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
        """Run forward propagation."""
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