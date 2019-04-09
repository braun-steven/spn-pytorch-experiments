from typing import List

import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import torch
from torch import distributions as dist
import torch.utils.data
from torch import nn
from torch import optim
from torch.nn import functional as F

from spn.structure.Base import Sum, Product, Node
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
import logging


logger = logging.getLogger(__name__)
EPS = 1e-8


class SPNClipper(object):
    """
    Clip SPN parameters back into valid space. The following parameter are being tracked:
    - Sum node weights: Assure weights sum to 1
    - Categorical probabilities: Assure probabilities sum to 1
    - Gaussian standard deviation: Assure positivity
    """

    def __init__(self, device, min_std=0.0001):
        """
        Create a SPNClipper.
        Args:
            device: Torch device.
            min_std: Minimum for the standard deviation.
        """
        self.min_std = torch.tensor(min_std).to(device)

    def __call__(self, module):
        if hasattr(module, "p"):
            p = module.p.data
            p.div_(torch.norm(p, 1, 0))

        if hasattr(module, "std"):
            std = module.std.data
            std.set_(torch.max(std, self.min_std))

        if hasattr(module, "weights"):
            weights = module.weights.data
            weights.set_(torch.softmax(weights, 0))


def children_to_torch(node: Node) -> List[nn.Module]:
    """
    Convert a SPN node's children to their torch equivalent.
    Args:
        node (Node): Input node.

    Returns:
        List[nn.Module]: List of children modules.
    """
    children = []

    # Collect children and convert them to torch nodes
    for ch in node.children:
        torch_cls = _node_to_torch_dict[ch.__class__]
        torch_child = torch_cls.from_spn(ch)
        children.append(torch_child)
    return children


class ProductNode(nn.Module):
    """A PyTorch product node module."""

    def __init__(self, children: List["SumNode"]):
        """
        Initialize ProductNode.
        
        Args:
            children (List[SumNode]): Product node children.
        """

        super(ProductNode, self).__init__()
        self.ch_nodes = nn.ModuleList(children)

    def forward(self, x):
        # Product is sum in logspace
        ch_res = [ch(x) for ch in self.ch_nodes]
        child_probs = torch.stack(ch_res, dim=1)
        res = torch.sum(child_probs, dim=1)
        return res.view(-1)

    @classmethod
    def from_spn(cls, node: Product):
        """
        Translate a SPN product node to a PyTorch product node implementation.

        Args:
            node (Product): Input product node.

        Returns:
            ProductNode: PyTorch product node implementation.
        """
        children = children_to_torch(node)
        torch_node = ProductNode(children=children)
        return torch_node


class SumNode(nn.Module):
    """
    A PyTorch sum node module.
    """

    def __init__(
        self, weights: List[float] = None, children: List["ProductNode"] = None
    ):
        """
        Create a SumNode

        Args:
            weights (List[float]): Input weights.
            children (List[ProductNode]): Product nodes children.
        """

        super(SumNode, self).__init__()
        self.ch_nodes = nn.ModuleList(children)

        # Initialize weights of not given
        if weights is None:
            weights = np.random.rand(len(self.ch_nodes))

        self.weights = nn.Parameter(torch.Tensor(weights))

    def forward(self, x):
        # Calculate children
        child_probs = torch.stack([ch(x) for ch in self.ch_nodes], dim=1)

        # Prepare weights
        log_weights = torch.log(self.weights)

        # Apply logsumexp
        probs = torch.logsumexp(child_probs + log_weights, dim=1)

        assert not torch.isnan(
            probs
        ).any(), "SumNode(weights={}) produced NaNs for input: \n{}\noutput\n{}".format(
            self.weights.data, child_probs, probs
        )
        return probs

    @classmethod
    def from_spn(cls, node: Sum):
        """
        Translate a SPN Sum node to a PyTorch Sum node implementation.

        Args:
            node (Sum): Input sum node.

        Returns:
            SumNode: PyTorch sum node implementation.
        """
        children = children_to_torch(node)
        torch_node = SumNode(weights=node.weights, children=children)
        return torch_node

    def __str__(self):
        return "SumNode(weights={})".format(self.weights.data.tolist())


class CategoricalNode(nn.Module):
    """A PyTorch categorical node module."""

    def __init__(self, num_categories: int, p: List[float] = None, scope: int = None):
        """
        Initialize the categorical node.

        Args:
            num_categories (int): Number of categories in this node.
            p (List[float]): Probabilities for each category.
        """
        super(CategoricalNode, self).__init__()
        assert scope is not None
        self.scope = scope

        # Generate uniform initial distribution if none was given
        if p is None:
            p = np.random.uniform(0, 1, num_categories)
            p = p / p.sum()

        # self.softmax = nn.Softmax()
        self.p = nn.Parameter(torch.Tensor(p))
        self.cat = dist.Categorical(self.p)

    def forward(self, x):
        # TODO: Project self.p back into valid space after gradient updates
        return self.cat.log_prob(x[:, self.scope])

    @classmethod
    def from_spn(cls, node: Categorical):
        """
        Create a PyTorch CategoricalNode from an SPN categorical node.

        Args:
            node (Categorical): Input node.

        Returns:
            ProductNode: PyTorch node of the given categorical node.
        """

        torch_node = CategoricalNode(
            num_categories=len(node.p), p=node.p, scope=node.scope
        )
        return torch_node

    def __str__(self):
        return "CategoricalNode(p={}, scope={})".format(
            self.p.data.tolist(), self.scope
        )


class GaussianNode(nn.Module):
    """A PyTorch gaussian node module."""

    def __init__(self, mean: float = None, std: float = None, scope: int = None):
        """
        Initialize the gaussiane node.

        Args:
            mean (float): Mean of the gaussian.
            std (float): Standard deviation of the gaussian.
            scope (int): Node scope over the data variables.
        """
        super(GaussianNode, self).__init__()
        assert scope is not None
        if std:
            assert (
                std > 0.0
            ), "Standard deviation must be larger than 0.0 but was " + str(std)
        self.scope = scope

        # Generate random mean and std if none were given
        if mean is None:
            mean = np.random.randn(1)
            std = np.random.rand(1)

        self.mean = nn.Parameter(torch.Tensor([mean]))
        self.std = nn.Parameter(torch.Tensor([std]))
        self.gauss = dist.Normal(loc=self.mean, scale=self.std)

    def forward(self, x):
        res = self.gauss.log_prob(x[:, self.scope])

        assert not torch.isnan(
            res
        ).any(), "GaussianNode(mean={}, std={}, scope={}) produced NaNs for input: \n{}\noutput: \n{}".format(
            self.mean.data, self.std.data, self.scope, x[:, self.scope], res
        )
        return res

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "GaussianNode(mean={}, std={}, scope={})".format(
            self.mean.data.item(), self.std.data.item(), self.scope
        )

    @classmethod
    def from_spn(cls, node: Gaussian):
        """
        Translate a SPN gaussian node into a PyTorch gaussian node.

        Args:
            node (Gaussian): Input gaussian node.

        Returns:
            GaussianNode: PyTorch Gaussian version of the input gaussian node.
        """

        torch_node = GaussianNode(mean=node.mean, std=node.stdev, scope=node.scope)
        return torch_node


class MultivariateGaussian(nn.Module):
    """A PyTorch multivariate gaussian node module."""

    def __init__(
        self, mean=None, covariance_surrogate=None, n_vars=None, scope: List[int] = None
    ):
        """
        Initialize the multivariate gaussiane node.

        Can be either initialized via given mean and covariance matrix or only by giving the number
        of variables which leads to random initialization.

        The covariance matrix is implicitly learned by learning a surrogate matrix X which constructs
        the covariance matrix via 0.5*(X + X') + eye(n)*n.

        Args:
            mean: Vector of means.
            covariance_surrogate: Covariance matrix surrogate X, that is Cov = XX^T.
            n_vars: Number of variables to model.
            scope (List[int]): Node scope over the data variables.
        """
        super(MultivariateGaussian, self).__init__()
        assert scope is not None

        if mean is not None:
            assert len(scope) == len(mean)

        if n_vars is not None:
            assert (
                len(scope) == n_vars
            ), "The scope has to be equal to the number of variables modeled with the multivariate gaussian."
            # assert (
            #     n_vars >= covariance_surrogate_dim
            # ), "The covariance surrogate dimension must be in 1 <= dim <= n_vars."

        # Check input
        if not ((n_vars is None) ^ (mean is None)):
            raise Exception(
                "Either n_vars or mean and covariance have to be given, but not none or both."
            )

        if (
            mean is not None
            and covariance_surrogate is None
            or mean is None
            and covariance_surrogate is not None
        ):
            raise Exception(
                "If mean is given, covariance matrix surrogate is necessary and vice versa."
            )

        self.scope = scope
        self.n_vars = n_vars

        # If mean is not set, generate random mean and covariance matrix surrogate
        if mean is None:
            mean = np.random.randn(n_vars)
            covariance_surrogate = np.random.randn(n_vars, n_vars)

        # Save mean and covariance matrix as parameter
        self.mean = nn.Parameter(torch.Tensor(mean))
        self.cov_surrogate = nn.Parameter(torch.rand(n_vars, n_vars))

    def forward(self, x):
        centered_surrogate = self.cov_surrogate - self.cov_surrogate.mean(dim=0)

        # Create a symmetric positive definite matrix
        # https://math.stackexchange.com/a/358092
        cov = 0.5 * (centered_surrogate + centered_surrogate.t())
        # OR: cov = torch.matmul(centered_surrogate, centered_surrogate.t())
        # since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
        #   is symmetric positive definite, which can be ensured by adding nI
        cov = cov + self.n_vars * torch.eye(self.n_vars)

        mvg = dist.MultivariateNormal(loc=self.mean, covariance_matrix=cov)
        res = mvg.log_prob(x[:, self.scope])

        assert not torch.isnan(
            res
        ).any(), "{} produced NaNs for input: \n{}\noutput: \n{}".format(
            self, x[:, self.scope], res
        )
        return res

    def __str__(self):
        return "Multinomial(mean={}, covariance_matrix={}, scope={})".format(
            self.mean.data, self.covariance_matrix.data, self.scope
        )

    def __repr__(self):
        return str(self)


class SPNLayer(nn.Module):
    """
    A PyTorch module that contains multiple SPNs with the same structure and treats them as single nodes in a layer.
    """

    def __init__(self, neuron: nn.Module, in_features: int, out_features: int):
        """
        Initialize the SPNLayer.

        Args:
            in_features (int): Number of input features for this layer.
            out_features (int): Number of output features for this layer.
        """
        super(SPNLayer, self).__init__()

        # Create out_features number of SPNNeurons
        neurons = [neuron(in_features) for _ in range(out_features)]
        self.spns = nn.ModuleList(neurons)

    def forward(self, x):
        # Feed forward each neuron and stack the results
        spn_results = [spn(x) for spn in self.spns]
        return torch.stack(spn_results, dim=1)


def ll_loss(model: nn.Module, data) -> torch.Tensor:
    """
    Default log likelihood loss.

    Args:
        model: PyTorch module.
        data: Input data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loss tensor and model output.
    """
    output = model(data[0])
    loss = -1 * torch.mean(output)
    return loss, output


def optimize_torch(
    model,
    data=None,
    loader=None,
    epochs=1000,
    batch_size=None,
    optimizer=None,
    loss_fn=ll_loss,
    device="cpu",
    seed=0,
) -> List[float]:
    """
    Optimize an SPN pytorch model.

    Args:
        model: SPN model as PyTorch module.
        data: Input data to optimize the log likelihood for.
        loader: Dataset loader. Can be set instead of data and batch_size.
        epochs: Number of epochs.
        batch_size: Data batch size.
        optimizer: Optimizer to use for gradient descent.
        loss_fn: Loss function that takes the data and the model output as input and outputs a loss tensor. 
            Defaults to ll_loss.
        device: Device to run the computation on.
        seed: Random number generator seed.

    Returns:
        List[float]: List of losses for epoch.

    """
    if loader is None and data is None:
        raise Exception("Either a loader or the data must be given.")

    if loader is not None and data is not None:
        raise Exception("Only loader or data can be specified.")

    # If loader is not given, create loader from data and batch_size arguments
    if loader is None:
        # If data is a list of inputs (e.g. [X, y]) stack them.
        if isinstance(data, list):
            tensors = [torch.tensor(d, dtype=torch.float32) for d in data]
            ds = torch.utils.data.TensorDataset(*tensors)
        else:
            tensor = torch.tensor(data, dtype=torch.float32)
            ds = torch.utils.data.TensorDataset(tensor)

        if batch_size is None:
            batch_size = len(ds)

        # Prepare training
        loader = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)

    # Set seed
    torch.manual_seed(seed)

    # Get correct device
    device = torch.device(device)

    # Get optimizer if none was set
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable training mode for model
    model.train()
    loss_list = []

    # Iterate number of epochs
    for i in range(epochs):
        running_loss = 0.0

        # Iterate dataset loader
        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                data = [d.to(device) for d in data]
            elif isinstance(data, torch.Tensor):
                data = data.to(device)
            else:
                raise Exception(
                    "Invalid data type: %s. Expected either torch.Tensor or list of torch.Tensor."
                    % type(data)
                )

            # Parameter optimization
            optimizer.zero_grad()
            loss, output = loss_fn(model, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        i,
                        batch_idx * loader.batch_size,
                        len(loader.dataset),
                        100.0 * batch_idx / len(loader),
                        loss.item(),
                    )
                )

        # Get loss per datapoint
        running_loss /= len(loader)
        loss_list.append(running_loss)

    return loss_list


_torch_to_node_dict = {
    SumNode: Sum,
    ProductNode: Product,
    GaussianNode: Gaussian,
    CategoricalNode: Categorical,
}
_node_to_torch_dict = {v: k for k, v in _torch_to_node_dict.items()}
