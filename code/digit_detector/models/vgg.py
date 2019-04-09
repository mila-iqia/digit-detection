'''
VGG11/13/16/19 in Pytorch.

Source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
'''

import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M',
              512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256,
              'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
              'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512,
              512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=7, classify=True):
        '''
        VGG11/13/16/19 model.

        Parameters
        ----------
        vgg_name : str
            The type of VGG. In ['VGG11', 'VGG13', 'VGG16', 'VGG19']
        num_classes : int
            Number of classes in the output of the model.
        classify : bool
            If True use it as a classifier otherwise use it as a
            a base network (feature extractor).

        '''
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

        self.classify = classify
        if self.classify:
            self.classifier = nn.Linear(512, num_classes)

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
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.classify:
            out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        '''
        Helper function to make layers in the model.

        Parameters
        ----------
        cfg : list
            list of layer type and output size. M is for MaxPooling and
            int are for output size.

        '''
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    '''Test the VGG class.'''
    print('Test VGG11')
    net = VGG('VGG11', num_classes=7)
    x = torch.randn(2, 3, 54, 54)
    y = net(x)
    print(y.size())

    print('Classify is False')
    net = VGG('VGG11', num_classes=7, classify=False)
    y = net(x)
    print(y.size())

    print('Test VGG13')
    net = VGG('VGG13', num_classes=7)
    y = net(x)
    print(y.size())

    print('Classify is False')
    net = VGG('VGG13', num_classes=7, classify=False)
    y = net(x)
    print(y.size())

    print('Test VGG16')
    net = VGG('VGG16', num_classes=7)
    y = net(x)
    print(y.size())

    print('Classify is False')
    net = VGG('VGG16', num_classes=7, classify=False)
    y = net(x)
    print(y.size())

    print('Test VGG19')
    net = VGG('VGG19', num_classes=7)
    y = net(x)
    print(y.size())

    print('Classify is False')
    net = VGG('VGG19', num_classes=7, classify=False)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()
