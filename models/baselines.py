import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self, num_classes=2):
        '''
        Convolutional Neural Network.

        Parameters
        ----------
        num_classes : int
            Number of classes for the output of the network.

        '''

        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(4608, num_classes)

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
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # Flatten based on batch size
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class BaselineCNN(nn.Module):  # Achieves ~91%

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
