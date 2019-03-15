'''
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        '''
        Basic block class for ResNet model. Basic block are usually
        used in ResNet 18 and 34.

        Parameters
        ----------
        in_planes : int
            Number of channels of the input of conv1 and optionaly in the
            conv2d of the shortcut.
        planes : int
            Number of channels produced by conv1.
        stride : int
            Stride to use for the conv2 and optionaly in the conv2d
            of the shortcut. Default = 1.

        '''
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        '''
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        out : ndarray
            Output to the network.

        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        '''
        Bottleneck block class for ResNet model. Bottleneck block are
        usually used in ResNet 50, 101 and 152.

        Parameters
        ----------
        in_planes : int
            Number of channels of the input of conv1 and optionaly in the
            conv2d of the shortcut.
        planes : int
            Number of channels produced by conv1.
        stride : int
            Stride to use for the conv2 and optionaly in the conv2d
            of the shortcut. Default = 1.

        '''
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        '''
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        out : ndarray
            Output to the network.

        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        '''
        Base class for ResNet.

        Parameters
        ----------
        block : nn.module
            nn.module of the type of block you want in the ResNet model.
            Option here are `BasicBlock` usually used in ResNet 18 and 34
            and `Bottleneck` block usually used in ResNet 50, 101 and 152.
        num_blocks : list
            List of 4 int containing the number of block per ResNet
            layer.
        num_classes: int
            Number of classes in the output of the model.

        '''
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        '''
        Base class for ResNet.

        Parameters
        ----------
        block : nn.module
            nn.module of the type of block you want in the ResNet model.
        planes : int
            Number of channels produced by conv1 of the block.
        num_blocks : int
            Number of block to have in the layer.
        stride : int
            Stride to use for the conv2 and optionaly in the conv2d
            of the shortcut for the block. Default = 1.

        Returns
        -------
        nn.sequential
            Sequential container containing the nn.modules of the different
            block in the layer.

        '''
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        out : ndarray
            Output to the network.

        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    '''
    ResNet18.

    Parameters
    ----------
    num_classes: int
        Number of classes in the output of the model.

    '''
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    '''
    ResNet34.

    Parameters
    ----------
    num_classes: int
        Number of classes in the output of the model.

    '''

    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    '''
    ResNet50.

    Parameters
    ----------
    num_classes: int
        Number of classes in the output of the model.

    '''

    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    '''
    ResNet101.

    Parameters
    ----------
    num_classes: int
        Number of classes in the output of the model.

    '''

    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    '''
    ResNet152.

    Parameters
    ----------
    num_classes: int
        Number of classes in the output of the model.

    '''

    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def test():
    '''Test the ResNet class.'''
    net = ResNet18(num_classes=7)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())
