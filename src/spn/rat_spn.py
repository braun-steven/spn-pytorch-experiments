import logging

import numpy as np
import spn.experiments.RandomSPNs.RAT_SPN as RAT_SPN
import spn.experiments.RandomSPNs.region_graph as region_graph
import tensorflow as tf
from spn.io.Graphics import plot_spn
import torch
from torch import nn
from torch.nn import functional as F
import time

from src.spn.distributions import Normal
from src.spn.distributions import IsotropicMultivariateNormal
from src.spn.distributions import MultivariateNormal
from src.spn.layers import Product, Sum
from src.utils.utils import time_delta_now

logger = logging.getLogger(__name__)


class RatSpnConstructor:
    def __init__(self, in_features, C, S, I, dropout=0.0):
        """
        RAT SPN class.

        Parameters according to the paper (see Args for descriptions).
        
        Args:
            in_features (int): Number of input features.
            C (int): Number of classes.
            S (int): Number of sum nodes.
            I (int): Number of distributions for each leaf node.
            dropout (float): Dropout probability.
        """
        self.in_features = in_features
        self.C = C
        self.S = S
        self.I = I
        self.dropout = dropout

        # Collect SPNs. Each random_split(...) call adds one SPN
        self._spns = []

    def _create_spn(self, num_parts, num_recursions=1):
        """Create an SPN from the given RAT parameters.
        
        Args:
            num_parts (int): Number of partitions.
            num_recursions (int, optional): Number of split repetitions. Defaults to 1.
        """
        # Build the SPN top down:
        layers = []

        for i in range(1, num_recursions + 1):
            prodlayer = RatProduct(in_features=num_parts ** i)
            is_last_layer = i == num_recursions
            if is_last_layer:
                # Output channels of distribution layer
                sum_in_channels = self.I
            else:
                # Output channels of product layer
                sum_in_channels = self.S ** 2

            sumlayer = Sum(
                in_channels=sum_in_channels,
                in_features=num_parts ** i,
                out_channels=self.S,
                dropout=self.dropout,
            )
            layers.append(prodlayer)
            layers.append(sumlayer)

        # Cardinality is the size of the region in the last partitions
        cardinality = np.ceil(self.in_features / (num_parts ** num_recursions)).astype(
            int
        )

        # prod = Product(
        #     in_features=self.in_features, cardinality=cardinality, randomize=True
        # )
        # gauss = Normal(
        #     multiplicity=self.I, in_features=self.in_features, dropout=self.dropout
        # )
        # layers.append(prod)
        # layers.append(gauss)

        # multivariate_gauss = MultivariateNormal(
        #     multiplicity=self.I,
        #     cardinality=cardinality,
        #     in_features=self.in_features,
        #     dropout=self.dropout,
        # )
        multivariate_gauss = IsotropicMultivariateNormal(
            multiplicity=self.I,
            cardinality=cardinality,
            in_features=self.in_features,
            dropout=self.dropout,
        )
        layers.append(multivariate_gauss)

        # Reverse layers, since they have to be defined bottom up
        layers = reversed(layers)

        # Define SPN by stacking the layers
        spn = nn.Sequential(*layers)
        print("Added spn:\n")
        print(spn)
        self._spns.append(spn)

    def random_split(self, num_parts, num_recursions=1):
        """Randomly split the region graph.
        
        Args:
            num_parts (int): Number of partitions.
            num_recursions (int, optional): Number of split repetitions. Defaults to 1.
        """

        if num_parts ** (num_recursions) > self.in_features:
            raise Exception(
                f"The values for num_parts ({num_parts}) and num_recursions ({num_recursions}) have to satisfiy the condition 'num_parts ** num_recursions ({num_parts ** num_recursions}) <= in_features ({self.in_features})'"
            )
        self._create_spn(num_parts, num_recursions)

    def build(self):
        """Build the RatSpn object from the defined region graph"""
        if len(self._spns) == 0:
            raise Exception(
                "No random split has been added. Call random_split(...) at least once before building the RatSpn."
            )
        return RatSpn(spns=self._spns, C=self.C, S=self.S)


class RatProduct(nn.Module):
    """
    Layerwise implementation of a RAT Product node.

    Builds the the combination of all children in two regions:
    res = []
    for n1 in R1, n2 in R2:
        res += [n1 * n2]
        
    TODO: Generalize to k regions.
    """

    def __init__(self, in_features, randomize=True):
        """
        Create a rat product node layer.

        Args:
            in_features (int): Number of input features.
            randomize (bool): Whether to randomize the selection of scopes.
                If false, scopes are chosen consecutively.
        """

        super().__init__()
        cardinality = 2
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
            torch.Tensor: Output of shape [batch, ceil(in_features/2), channel * channel].
        """

        if self._pad:
            # Pad marginalized node
            x = F.pad(x, pad=[0, 0, 0, 1], mode="constant", value=0.0)

        # Create zero tensor and sum up afterwards
        batch = x.shape[0]
        channels = x.shape[2]
        result = torch.zeros(batch, self._out_features, channels).to(x.device)

        # Build outer sum, using broadcasting, this can be done with
        # modifying the tensor timensions:
        # left: [n, d, c] -> [n, d, c, 1]
        # right: [n, d, c] -> [n, d, 1, c]
        # left + right with broadcasting: [n, d, c, 1] + [n, d, 1, c] -> [n, d, c, c]
        left = x[:, self._scopes[0, :], :].unsqueeze(3)
        right = x[:, self._scopes[1, :], :].unsqueeze(2)
        result = left + right

        # Put the two channel dimensions from the outer sum into one single dimension:
        # [n, d/2, c, c] -> [n, d/2, c * c]
        result = result.view(*result.shape[:-2], result.shape[-1] ** 2)
        return result

    def __repr__(self):
        return "RatProduct(in_features={}, cardinality={})".format(
            self.in_features, self.cardinality
        )


class RatSpn(nn.Module):
    """
    RAT SPN PyTorch implementation with layerwise tensors.

    See also:
    https://arxiv.org/abs/1806.01910
    """

    def __init__(self, spns, C, S):
        """
        Initialize the RAT SPN  PyTorch module.

        Args:
            spns: Internal SPNs which correspond to random region splits.
            C: Number of classes.
            S: Number of sum nodes at each sum layer.
        """
        super().__init__()
        self.C = C
        self._priors = np.log(1 / self.C)
        self.spns = nn.ModuleList(spns)
        self._root = Sum(in_channels=len(spns) * S ** 2, in_features=1, out_channels=C)

    def forward(self, x):
        """Computes the posterior probabilities P(X | C) for each class."""
        x = [spn(x) for spn in self.spns]
        x = [res.squeeze(1) for res in x]
        x = torch.stack(x, dim=1)

        # Merge results from the different SPN into the channel dimension
        x = x.view(x.shape[0], 1, -1)

        # Apply C sum node outputs
        x = self._root(x)
        x = x.squeeze(1)

        # Compute P(C | X) = P(X | C) * P(C) / Z
        # where Z = sum_i { P(X | C = c_i) * P(C = C_i) }
        # and P(X | C) is the output of the root
        z = torch.logsumexp(x + self._priors, dim=1).unsqueeze(-1)

        # Posterior in log domain
        # logP(C | X) = logP(X | C) + logP(C) - logsumexp(logP(X | C) + logP(C))
        posteriors = x + self._priors - z
        return posteriors

    def __repr__(self):
        spns_str = ""
        for i in range(len(self.spns)):
            spns_str += f"(Split-{i}): {self.spns[i]}\n"

        return f"RatSpn(C={self.C}, spns=\n{spns_str})"


if __name__ == "__main__":
    # rg = region_graph.RegionGraph(range(20))
    # for _ in range(0, 3):
    #     rg.random_split(2, 2)
    # args = RAT_SPN.SpnArgs()
    # args.normalized_sums = True
    # args.num_sums = 2
    # args.num_gauss = 2
    d = 128
    b = 64
    rg = RatSpnConstructor(in_features=d, C=10, S=2, I=2, dropout=0.1)
    for i in range(10):
        rg.random_split(2, 2)
    rat = rg.build()
    import torch

    x = torch.randn(b, d)
    x = rat(x)
    print(x)
    print(x.exp())
    print(x.exp().sum(1))
    print(x.shape)
