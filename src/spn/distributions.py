"""
Module that contains a set of distributions with learnable parameters.
"""
from abc import ABC
import logging
import torch
from torch import nn
from torch import distributions as dist

logger = logging.getLogger(__name__)


def dist_forward(distribution, x):
    """
    Forward pass with an arbitrary PyTorch distribution.

    Args:
        distribution: PyTorch base distribution which is used to compute the log probabilities of x.
        x: Input to compute the log probabilities of.
           Shape [b, d, c].

    Returns:
        torch.Tensor: Log probabilities for each feature.
    """
    # Make room for multiplicity of layer
    # Output shape: [b, d, c, 1]
    x = x.unsqueeze(3)

    # Compute gaussians
    # Output shape: [b, d, c, multiplicity]
    x = distribution.log_prob(x)

    # Merge in_channel and multiplicity axis (assumes equal scopes across the in_channel
    # dimension (e.g. color channels))
    batch_size = x.shape[0]
    in_features = x.shape[1]

    # Output shape: [b, d, multiplicity * c]
    x = x.view(batch_size, in_features, -1)
    return x


class Leaf(nn.Module, ABC):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.
    """

    def __init__(self, multiplicity, in_channels, in_features):
        """
        Create the leaf layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_channels: Number of input channels.
            in_features: Number of input features.
        """
        super(Leaf, self).__init__()
        assert multiplicity > 0, "Multiplicity must be > 0 but was %s." % multiplicity
        self._multiplicity = multiplicity
        self._in_channels = in_channels
        self._in_features = in_features


class Normal(Leaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, multiplicity, in_channels, in_features):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_channels: Number of input channels.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_channels, in_features)

        # Create gaussian means and stds
        self.means = nn.Parameter(
            torch.randn(1, in_features, in_channels, multiplicity)
        )
        self.stds = nn.Parameter(torch.rand(1, in_features, in_channels, multiplicity))
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

    def forward(self, x):
        return dist_forward(self.gauss, x)


class Beta(Leaf):
    """Beta layer. Maps each input feature to its beta log likelihood."""

    def __init__(self, multiplicity, in_channels, in_features):
        """Creat a beta layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_channels: Number of input channels.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_channels, in_features)

        # Create beta parameters
        self.concentration0 = nn.Parameter(
            torch.rand(1, in_features, in_channels, multiplicity)
        )
        self.concentration1 = nn.Parameter(
            torch.rand(1, in_features, in_channels, multiplicity)
        )
        self.beta = dist.Beta(
            concentration0=self.concentration0, concentration1=self.concentration1
        )

    def forward(self, x):
        return dist_forward(self.beta, x)


class Cauchy(Leaf):
    """Cauchy layer. Maps each input feature to cauchy beta log likelihood."""

    def __init__(self, multiplicity, in_channels, in_features):
        """Creat a cauchy layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_channels: Number of input channels.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_channels, in_features)
        self.means = nn.Parameter(
            torch.randn(1, in_features, in_channels, multiplicity)
        )
        self.stds = nn.Parameter(torch.rand(1, in_features, in_channels, multiplicity))
        self.cauchy = dist.Cauchy(loc=self.means, scale=self.stds)

    def forward(self, x):
        return dist_forward(self.cauchy, x)


class Chi2(Leaf):
    """Chi square distribution layer"""

    def __init__(self, multiplicity, in_channels, in_features):
        """Creat a chi square layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_channels: Number of input channels.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_channels, in_features)
        self.df = nn.Parameter(torch.rand(1, in_features, in_channels, multiplicity))
        self.chi2 = dist.Chi2(df=self.df)

    def forward(self, x):
        return dist_forward(self.chi2, x)


class Gamma(Leaf):
    """Gamma distribution layer."""

    def __init__(self, multiplicity, in_channels, in_features):
        """Creat a chi square layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_channels: Number of input channels.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_channels, in_features)
        self.concentration = nn.Parameter(
            torch.rand(1, in_features, in_channels, multiplicity)
        )
        self.rate = nn.Parameter(1, in_features, in_channels, multiplicity)
        self.gamma = dist.Gamma(concentration=self.concentration, rate=self.rate)

    def forward(self, x):
        return dist_forward(self.gamma, x)


class Representations(Leaf):
    def __init__(self, distributions, multiplicity, in_channels, in_features):
        """
        Create a layer that stack multiple representations of a feature along the scope dimension.

        Args:
            distributions: List of possible distributions to represent the feature with.
            multiplicity: Multiplicity of how many nodes each distribution is assigned to.
            in_channels: Number of input channels.
            in_features: Number of input features.
        """
        super().__init__(multiplicity, in_channels, in_features)
        reprs = [
            distr(multiplicity, in_channels, in_features) for distr in distributions
        ]
        self.representations = nn.ModuleList(reprs)

    def forward(self, x):
        results = [d(x) for d in self.representations]

        # Stack along output channel dimension
        x = torch.cat(results, dim=1)
        return x
