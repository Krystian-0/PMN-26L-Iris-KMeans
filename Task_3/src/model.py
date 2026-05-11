import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self, kernel_size=3, pool_type='max'):
        super(CustomCNN, self).__init__()
        self.pool_type = pool_type.lower()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)

        if self.pool_type == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self._apply_pooling(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self._apply_pooling(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self._apply_pooling(x)

        x = x.view(-1, 128 * 4 * 4)
        features = x

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if return_features:
            return x, features
        return x

    def _apply_pooling(self, x):
        if self.pool_type == 'min':
            return -self.pool(-x)
        elif self.pool_type == 'none':
            return x
        else:
            return self.pool(x)