import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class Sum(nn.Module):
    def __init__(self, in_channels, in_features, out_channels, dropout=0.0):
        super(Sum, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_channels = out_channels
        self.dropout = dropout
        assert out_channels > 0, (
            "Number of output channels must be at least 1, but was %s." % out_channels
        )
        in_features = int(in_features)
        # Weights, such that each sumnode has its own weights
        ws = torch.rand(1, in_features, in_channels, out_channels)
        # Normalize over in_channel xis
        F.normalize(ws, p=1, dim=2, out=ws)
        self.sum_weights = nn.Parameter(ws)
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=dropout)

    def forward(self, x):
        """
        Sum layer foward pass.

        Args:
            x: Input of shape [batch, in_features, in_channels].

        Returns:
            torch.Tensor: Output of shape [batch, in_features, out_channels]
        """
        # assert x.dim() == 3

        # Apply dropout: Set random sum node children to 0 (-inf in log domain)
        if self.dropout > 0.0:
            r = self._bernoulli_dist.sample(x.shape).type(torch.bool)
            x[r] = np.NINF

        # Multiply x with weights in logspace
        # Resuts in shape: [n, d, ic, oc]
        x = x.unsqueeze(3) + torch.log(torch.softmax(self.sum_weights, dim=2))

        # Compute sum via logsumexp along dimension "ic" (in_channels)
        # Resuts in shape: [n, d, oc]
        x = torch.logsumexp(x, dim=2)

        return x

    def __repr__(self):
        return "Sum(in_channels={}, in_features={}, out_channels={})".format(
            self.in_channels, self.in_features, self.out_channels
        )


class Product(nn.Module):
    """
    Product Node Layer that chooses k scopes as children for a product node.
    """

    def __init__(self, in_features, cardinality, randomize=False):
        """
        Create a product node layer.

        Args:
            in_features (int): Number of input features.
            cardinality (int): Number of random children for each product node.
            randomize (bool): Whether to randomize the selection of scopes.
                If false, scopes are chosen consecutively.
        """

        super(Product, self).__init__()
        self.in_features = in_features
        self.cardinality = cardinality

        in_features = int(in_features)
        self._cardinality = cardinality
        # Check if forward pass needs padding
        self._pad = in_features % cardinality != 0
        self._out_features = np.ceil(in_features / cardinality).astype(int)

        # Collect scopes for each product child
        self._scopes = [[] for _ in range(cardinality)]
        # Create random sequence of scopes
        if randomize:
            scopes = np.random.permutation(in_features)
        else:
            scopes = range(in_features)

        # For two consecutive (random) scopes
        for i in range(0, in_features, cardinality):
            for j in range(cardinality):
                if i + j < in_features:
                    self._scopes[j].append(scopes[i + j])
                else:
                    # Case: d mod cardinality != 0 => Create marginalized nodes with prob 1.0
                    # Pad x in forward pass on the right: [n, d, c] -> [n, d+1, c] where index
                    # d+1 is the marginalized node (index "in_features")
                    self._scopes[j].append(in_features)

        # Transform into numpy array for easier indexing
        self._scopes = np.array(self._scopes)

    def forward(self, x):
        """
        Product layer forward pass.

        Args:
            x: Input of shape [batch, in_features, channel].

        Returns:
            torch.Tensor: Output of shape [batch, ceil(in_features/cardinality), channel].
        """
        # assert x.dim() == 3

        if self._pad:
            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 1], mode="constant", value=0.0)

        # Create zero tensor and sum up afterwards
        batch = x.shape[0]
        channels = x.shape[2]
        result = torch.zeros(batch, self._out_features, channels).to(x.device)

        # Build the product between all children
        for j in range(self._cardinality):
            result += x[:, self._scopes[j, :], :]
        return result

    def __repr__(self):
        return "Product(in_features={}, cardinality={})".format(
            self.in_features, self.cardinality
        )


if __name__ == "__main__":
    from src.spn.distributions import Normal
    from src.spn.layers import Sum, Product
    import torch
    from torch import nn

    # 1 Sample, 4 features, 1 channel
    x = torch.rand(1, 4, 1)

    p = Product(in_features=4, cardinality=2, randomize=False)
    p(x)
