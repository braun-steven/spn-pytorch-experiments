"""
Module that contains a set of distributions with learnable parameters.
"""
from abc import ABC
import logging
import torch
import numpy as np
from torch import nn
from torch import distributions as dist
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def dist_forward(distribution, x):
    """
    Forward pass with an arbitrary PyTorch distribution.

    Args:
        distribution: PyTorch base distribution which is used to compute the log probabilities of x.
        x: Input to compute the log probabilities of.
           Shape [n, d].

    Returns:
        torch.Tensor: Log probabilities for each feature.
    """
    # Make sure, that the input has the correct dimensions: [n, d]
    # assert x.dim() == 2

    # Make room for multiplicity of layer
    # Output shape: [n, d, 1]
    x = x.unsqueeze(2)

    # Compute gaussians
    # Output shape: [n, d, multiplicity]
    x = distribution.log_prob(x)

    return x


class Leaf(nn.Module, ABC):
    """
    Abstract layer that maps each input feature into a specified
    representation, e.g. Gaussians.

    Implementing layers shall be valid distributions.
    """

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """
        Create the leaf layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.
            droptout: Dropout probabilities.
        """
        super(Leaf, self).__init__()
        assert multiplicity > 0, "Multiplicity must be > 0 but was %s." % multiplicity
        self.multiplicity = multiplicity
        self.in_features = in_features
        self.dropout = dropout
        # self._bernoulli_dist = dist.Bernoulli(probs=dropout)

    def forward(self, x):
        # Apply dropout sampled from a bernoulli
        if self.dropout > 0.0:
            x = x * self._bernoulli_dist.sample(x.shape)
        return x


class Normal(Leaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, in_features, multiplicity))
        self.stds = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

    def forward(self, x):
        x = dist_forward(self.gauss, x)
        x = super().forward(x)
        return x


class MultivariateNormal(Leaf):
    """Multivariate Gaussian layer."""

    def __init__(self, multiplicity, in_features, cardinality, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.
            cardinality: Number of features covered.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.cardinality = cardinality
        self._pad_value = in_features % cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)
        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        # Create gaussian means and covs
        self.means = nn.Parameter(
            torch.randn(multiplicity * self._n_dists, cardinality)
        )

        # Generate covariance matrix via the cholesky decomposition: s = A'A where A is a triangular matrix
        # Further ensure, that diag(a) > 0 everywhere, such that A has full rank
        rand = torch.rand(multiplicity * self._n_dists, cardinality, cardinality)

        # Make a matrices triangular
        for i in range(multiplicity * self._n_dists):
            rand[i, :, :].tril_()

        self.triangular = nn.Parameter(rand)
        self._mv = dist.MultivariateNormal(loc=self.means, scale_tril=self.triangular)

    def forward(self, x):
        # Pad dummy variable via reflection
        if self._pad_value != 0:
            x = F.pad(x, pad=[0, 0, 0, self._pad_value], mode="reflect")

        # Make room for multiplicity of layer
        # Output shape: [n, 1, d]
        batch_size = x.shape[0]
        x = x.view(batch_size, self._n_dists, self.cardinality)
        x = x.repeat(1, self.multiplicity, 1)

        # Compute multivariate gaussians
        # Output shape: [n, multiplicity, d / cardinality]
        __import__("ipdb").set_trace(context=13)
        x = self._mv.log_prob(x)

        # Output shape: [n, d / cardinality, multiplicity]
        x = x.view(batch_size, self._n_dists, self.multiplicity)

        # Apply dropout defined in super class
        x = super().forward(x)

        return x


class Beta(Leaf):
    """Beta layer. Maps each input feature to its beta log likelihood."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a beta layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)

        # Create beta parameters
        self.concentration0 = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.concentration1 = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.beta = dist.Beta(
            concentration0=self.concentration0, concentration1=self.concentration1
        )

    def forward(self, x):
        x = dist_forward(self.beta, x)
        x = super().forward(x)
        return x


class Cauchy(Leaf):
    """Cauchy layer. Maps each input feature to cauchy beta log likelihood."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a cauchy layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_channels: Number of input channels.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.means = nn.Parameter(torch.randn(1, in_features, multiplicity))
        self.stds = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.cauchy = dist.Cauchy(loc=self.means, scale=self.stds)

    def forward(self, x):
        x = dist_forward(self.cauchy, x)
        x = super().forward(x)
        return x


class Chi2(Leaf):
    """Chi square distribution layer"""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a chi square layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.df = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.chi2 = dist.Chi2(df=self.df)

    def forward(self, x):
        x = dist_forward(self.chi2, x)
        x = super().forward(x)
        return x


class Gamma(Leaf):
    """Gamma distribution layer."""

    def __init__(self, multiplicity, in_features, dropout=0.0):
        """Creat a chi square layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.concentration = nn.Parameter(torch.rand(1, in_features, multiplicity))
        self.rate = nn.Parameter(1, in_features, multiplicity)
        self.gamma = dist.Gamma(concentration=self.concentration, rate=self.rate)

    def forward(self, x):
        x = dist_forward(self.gamma, x)
        x = super().forward(x)
        return x


class Representations(Leaf):
    def __init__(self, distributions, multiplicity, in_features, dropout=0.0):
        """
        Create a layer that stack multiple representations of a feature along the scope dimension.

        Args:
            distributions: List of possible distributions to represent the feature with.
            multiplicity: Multiplicity of how many nodes each distribution is assigned to.
            in_features: Number of input features.
        """
        super().__init__(multiplicity, in_features, dropout)
        reprs = [distr(multiplicity, in_features, dropout) for distr in distributions]
        self.representations = nn.ModuleList(reprs)

    def forward(self, x):
        results = [d(x) for d in self.representations]

        # Stack along output channel dimension
        x = torch.cat(results, dim=1)
        return x


class IsotropicMultivariateNormal(Leaf):
    """Isotropic multivariate gaussian layer.

    The covariance is simplified to:

    cov = sigma^2 * I

    Maps k input feature to their multivariate gaussian log likelihood."""

    def __init__(self, multiplicity, cardinality, in_features, dropout=0.0):
        """Creat a gaussian layer.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            cardinality: Number of features per gaussian.
            in_features: Number of input features.

        """
        super().__init__(multiplicity, in_features, dropout)
        self.cardinality = cardinality

        # Number of different distributions: total number of features
        # divided by the number of features in each gaussian

        self._pad_value = in_features % cardinality
        self._out_features = np.ceil(in_features / cardinality).astype(int)

        self._n_dists = np.ceil(in_features / cardinality).astype(int)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(multiplicity, self._n_dists, cardinality))
        self.stds = nn.Parameter(torch.rand(multiplicity, self._n_dists, cardinality))
        self.cov_factors = nn.Parameter(
            torch.zeros(multiplicity, self._n_dists, cardinality, 1),
            requires_grad=False,
        )
        self.gauss = dist.LowRankMultivariateNormal(
            loc=self.means, cov_factor=self.cov_factors, cov_diag=self.stds
        )

        # Fix LowRankMultivariateNormal elements
        # self.gauss.cov_diag.data.set_(self.std)

    def forward(self, x):
        # Pad dummy variable via reflection
        if self._pad_value != 0:
            # Do unsqueeze and squeeze due to padding not being allowed on 2D tensors
            x = x.unsqueeze(1)
            x = F.pad(x, pad=[0, self._pad_value // 2], mode="reflect")
            x = x.squeeze(1)

        # Make room for multiplicity of layer
        # Output shape: [n, 1, d]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self._n_dists, self.cardinality)

        # Compute multivariate gaussians
        # Output shape: [n, multiplicity, d / cardinality]
        x = self.gauss.log_prob(x)

        # Output shape: [n, d / cardinality, multiplicity]
        x = x.permute((0, 2, 1))

        # Apply dropout defined in super class
        x = super().forward(x)
        return x


if __name__ == "__main__":
    bs = 10
    n_features = 6
    cardinality = 2
    n_dists = n_features // cardinality
    multiplicity = 5

    # loc = torch.randn(4)
    # triang = torch.rand(4, 4).tril_()
    # cov = triang.t().matmul(triang)
    loc = torch.tensor([1.0, 0.5])
    # cov = torch.tensor([[1.0, 0.7], [0.7, 2.0]])
    triang = torch.tensor([[1.0, 0.0], [0.7, 2.0]])
    mv = dist.MultivariateNormal(loc=loc, scale_tril=triang)
    # mv = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)

    model = MultivariateNormal(multiplicity=10, in_features=2, cardinality=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):

        model.train()

        loss_fn = nn.NLLLoss()
        for batch_idx in range(1000):
            # Send data to correct device
            data = mv.sample([10])

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data)

            # Comput loss
            loss = -1 * output.mean()

            # Backprop
            loss.backward()
            optimizer.step()

            # Log stuff
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        100 * 1000,
                        100.0 * batch_idx / 1000,
                        loss.item() / 100,
                    )
                )
                print(model.means)
                # print(model.triangular.permute(0, 2, 1).matmul(model.triangular))
                print(model.triangular)

        print(model.means)
