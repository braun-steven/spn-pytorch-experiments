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
    - Linear Layer: n_in x 20
    - Linear Layer: 20 x 5
    - Linear Layer: 5 x 1
    """

    def __init__(self, in_features):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 10)
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
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 20)
        self.spn1 = SPNLayer(
            neuron=RandomSubspaceNeuron, in_features=20, out_features=20
        )
        self.bn1 = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, 10)
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
        x = F.relu(self.fc2(x))  # Second linear layer
        x = self.spn1(x)  # SPN

        # SPNs return log likelihoods in the range of -inf to 0
        # Map -inf to 0 to something sane where a higher activation
        # is equal to a high probability (lls near 0).
        x = torch.log(-x)  # Squash lls between 0 and some positive value

        # Map high activation to high probability (since torch.log(-x) has
        # inverted this correlation)
        # TODO: Search for alternatives | e.g. 1/x, but produces NaNs
        x = torch.max(x) - x
        # x = self.bn1(x)  # TODO: Check out effect of batch norm
        x = self.fc3(x)  # Classification layer
        return F.log_softmax(x, dim=1)


class TestNode(nn.Module):
    def __init__(self, in_features, n_mv=2):
        # Init
        super(TestNode, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=1)


class RandomSubspaceNeuron(nn.Module):
    def __init__(self, in_features, n_mv=2):
        # Init
        super(RandomSubspaceNeuron, self).__init__()

        scopes = torch.randperm(in_features)
        scopes = np.random.permutation(in_features)

        sums = []

        for i in range(0, in_features, 2):
            scope_1 = scopes[i]
            scope_2 = scopes[i + 1]

            # Create n_mv MultivariateGaussian from these two scopes
            mvs = []
            for _ in range(n_mv):
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
        return self.root(x)


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
