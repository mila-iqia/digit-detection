import torch
import torch.nn as nn

from vgg import VGG


class FC_Layer(nn.Module):

    '''
    Base class that uses a single fully connected layer to
    calculate the multiloss functions of the network
    '''

    def __init__(self, in_features, out_features):
        '''
        n_digits module. This network predicts the number
        of digits in an image.


        Parameters
        ----------
        n_input : int
            number of
        '''

        super(FC_Layer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):

        x = self.fc(x)

        return x


class MultiLoss(nn.Module):
    '''

    base_net : a nn.Module instance with a forward method implemented.

    FCLayer : a nn.Module class for the fully connected layers.
    '''
    def __init__(self, base_net, FCLayer):

        super(MultiLoss, self).__init__()
        self.base_net = base_net

        self.in_dim = 512

        self.fc_layers = torch.nn.ModuleList()

        out_dims = [7, 10, 10, 10, 10, 10]
        for out_dim in out_dims:

            fc_layer = FCLayer(self.in_dim, out_dim)
            self.fc_layers.append(fc_layer)

    def forward(self, x):

        preds = []
        features = self.base_net(x)

        # Per branch prediction
        for fc_layer in self.fc_layers:
            preds.append(fc_layer(features))

        return preds


def test_multiloss():
    base_net = VGG('VGG11', classify=False)

    model = MultiLoss(base_net, FC_Layer)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    print(y)


if __name__ == "__main__":
    test_multiloss()
    print("Success!")
