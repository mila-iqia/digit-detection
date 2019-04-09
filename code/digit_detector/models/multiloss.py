import torch
import torch.nn as nn


class FC_Layer(nn.Module):

    def __init__(self, in_features, out_features):
        '''
        Base class that uses a single fully connected layer to
        calculate the multiloss functions of the network.

        Parameters
        ----------
        in_features : int
            Number of of input features. (e.g., size of the feature map)
        out_features : int
            Number of output features. (e.g., number of classes)

        '''

        super(FC_Layer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

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

        x = self.fc(x)

        return x


class MultiLoss(nn.Module):

    def __init__(self, base_net, FCLayer, in_dim_fclayer):
        '''
        Correspond to the n_digits module. This network predicts the
        number of digits in an image.

        Parameters
        ----------
        base_net : nn.Module
            Instance network with a forward method implemented.
        FCLayer : nn.Module
            Class network for the fully connected layers.
        in_dim_fclayer : int
            Size of the feature map (i.e., input to feed to the FCLayer)

        '''

        super(MultiLoss, self).__init__()
        self.base_net = base_net

        self.in_dim = in_dim_fclayer

        self.fc_layers = torch.nn.ModuleList()

        # corresponds to the n_digits module.
        out_dims = [7, 10, 10, 10, 10, 10]
        for out_dim in out_dims:
            fc_layer = FCLayer(self.in_dim, out_dim)
            self.fc_layers.append(fc_layer)

    def forward(self, x):
        '''
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        preds : list
            Output to the network. List of ndarray containing the
            predictions of the different tasks.

        '''

        preds = []
        features = self.base_net(x)

        # Per branch prediction
        for fc_layer in self.fc_layers:
            preds.append(fc_layer(features))

        return preds
