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
        self.spn1 = SPNLayer(neuron=SPNNeuron, in_features=20, out_features=20)
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


class SPNNeuron(nn.Module):
    """A PyTorch neuron that implements a full SPN."""

    def __init__(self, in_features: int):
        """
        Initialize an SPNNeuron.

        Args:
            in_features (int): Number of input features.
        """
        super(SPNNeuron, self).__init__()
        self.in_features = in_features

        # List of gaussians
        gs0 = []
        gs1 = []

        # Generate random means and stdevs
        means = np.random.randn(in_features, 3)
        stdevs = np.random.rand(in_features, 3)

        # Produce two different gaussians for each input
        for i in range(in_features):
            g0 = GaussianNode(mean=means[i, 0], std=stdevs[i, 0], scope=i)
            g1 = GaussianNode(mean=means[i, 1], std=stdevs[i, 1], scope=i)

            # Collect gaussians
            gs0.append(g0)
            gs1.append(g1)

        # Collect prod nodes
        # It holds:
        # scope(prod0[i]) == scope(prod1[i])
        prods0 = []
        prods1 = []

        # Build all first order combinations of gaussians from first group
        for g_a in gs0:
            for g_b in gs0:
                prods0.append(ProductNode(children=[g_a, g_b]))

        # Build all first order combinations of gaussians from second group
        for g_a in gs1:
            for g_b in gs1:
                prods1.append(ProductNode(children=[g_a, g_b]))

        # Collect sum nodes
        sums = []
        n_sum_nodes = in_features ** 2
        weights = np.random.rand(n_sum_nodes, 2)
        weights = weights / weights.sum(axis=1, keepdims=True)
        for i in range(n_sum_nodes):
            sum_node = SumNode(weights=weights[i, :], children=[prods0[i], prods1[i]])
            sums.append(sum_node)

        # Construct root node as product of all sums
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
