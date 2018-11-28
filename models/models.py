import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):

    # placeholder CNN
    def __init__(self):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7744, 4096)
        self.fc2 = nn.Linear(4096, 7)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten based on batch size

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
