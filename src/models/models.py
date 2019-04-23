"""
Contains classes for the comparison of models on the MNIST dataset.
Main models:
- MLPNet: Feed forward NN with linear layers
- SPNNet: Same as MLPNet but replaces certain layers with SPNLayer
- SPNNeuron: Defines the SPN architecture of a single neuron in a SPNLayer
"""

import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from src.models.pytorch import SPNLayer
from src.models.pytorch import GaussianNode
from src.models.pytorch import ProductNode
from src.models.pytorch import SumNode
from src.models.pytorch import MultivariateGaussian


class MLPNet(nn.Module):
    """
    Network architecture:
    - Linear Layer: n_in x 128
    - Linear Layer: 128 x 20
    - Linear Layer: 20 x 20
    - Linear Layer: 20 x 10
    """

    def __init__(self, in_features):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        self.fc4 = nn.Linear(10, 10)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class SPNNet(nn.Module):
    """
    Network architecture:
    - Linear Layer: n_in x 20
    - SPN Layer: 20 x 5
    - Batchnorm
    - Linear Layer: 5 x 1
    """

    def __init__(self, in_features):
        super(SPNNet, self).__init__()

        # Define Layers
        self.fc1 = nn.Linear(in_features, 32)
        # self.fc2 = nn.Linear(128, 20)
        self.spn1 = SPNLayer(neuron=SPNNeuron, in_features=32, out_features=16)
        self.spn2 = SPNLayer(neuron=SPNNeuron, in_features=16, out_features=10)
        self.fc2 = nn.Linear(10, 10)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Reshape height and width into single dimension
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))  # First linear layer
        x = self.spn1(x)  # SPN
        x = self.spn2(x)  # SPN
        x = self.fc2(x)  # Linear
        return F.log_softmax(x, dim=1)


class SPNNeuron(nn.Module):
    def __init__(self, in_features, n_mv=2):
        """
        Initialize the SPNNeuron.

        Args:
            in_features: Number of input features.
            n_mv: Number of different pairwise independence mixtures of the leaf nodes.
        """
        # Init
        super(SPNNeuron, self).__init__()

        # Create random sequence of scopes
        scopes = np.random.permutation(in_features)

        sums = []

        # For two consecutive (random) scopes
        for i in range(0, in_features, 2):
            scope_1 = scopes[i]
            scope_2 = scopes[i + 1]

            # Create n_mv MultivariateGaussian from these two scopes
            mvs = []
            for _ in range(n_mv):
                # TODO: MVG are currently not trainable
                # mv = MultivariateGaussian(n_vars=2, scope=[scope_1, scope_2])
                # mvs.append(mv)

                g1 = GaussianNode(scope=scope_1)
                g2 = GaussianNode(scope=scope_2)

                prod = ProductNode([g1, g2])
                mvs.append(prod)

            sumnode = SumNode(children=mvs)
            sums.append(sumnode)

        self.root = ProductNode(children=sums)

    def forward(self, x):
        x = self.root(x)
        return x


def get_model_by_tag(tag: str, device) -> nn.Module:
    """
    Return the model for a given tag.

    Args:
        tag (str): Model tag.
        device: Device to create the model on.

    Returns:
        nn.Module: PyTorch model.
    """
    # Select model
    if tag.lower() == "spn":
        model = SPNNet(in_features=28 * 28).to(device)
    elif tag.lower() == "mlp":
        model = MLPNet(in_features=28 * 28).to(device)
    else:
        raise Exception("Invalid network: %s" % tag)

    return model
