import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self, kernel_size=3, pool_type='max', num_classes=10):
        super(CustomCNN, self).__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding)

        self.pool_type = pool_type
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

        self.dropout = nn.Dropout(0.3)

        self._to_linear = None
        self._dummy_forward(torch.zeros(1, 3, 32, 32))

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _dummy_forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        features = x

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)

        if return_features:
            return out, features
        return out