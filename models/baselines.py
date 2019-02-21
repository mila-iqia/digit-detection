import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):

    def __init__(self, num_classes):
        '''
        Placeholder CNN
        '''
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7744, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        '''
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        x : ndarray
            Output to the network.

        '''

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten based on batch size
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class BaselineCNN_dropout(nn.Module):

    def __init__(self, num_classes, p=0.5):
        '''
        Placeholder CNN
        '''
        super(BaselineCNN_dropout, self).__init__()

        self.p = p
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(self.p)

        self.fc1 = nn.Linear(7744, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        '''
        Forward path.
        Parameters
        ----------
        x : ndarray
            Input to the network.
        Returns
        -------
        x : ndarray
            Output to the network.
        '''

        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        # Flatten based on batch size
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
