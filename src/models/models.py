"""
Contains classes for the comparison of models on the MNIST dataset.
Main models:
- MLPNet: Feed forward NN with linear layers
- SPNNet: Same as MLPNet but replaces certain layers with SPNLayer
- SPNNeuron: Defines the SPN architecture of a single neuron in a SPNLayer
"""
import logging

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

from src.models.pytorch import GaussianNode, ProductNode, SumNode
from src.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from src.spn.distributions import Normal
from src.spn import layers

logger = logging.getLogger(__name__)

###########
# Neurons #
###########


class SPNNeuronVectorized(nn.Module):
    """
    SPN Neuron implementation using the vectorized layers.
    """

    def __init__(self, in_features, n_gaussians=3):
        """
        Initialize the SPNNeuron.

        Args:
            in_features: Number of input features.
            n_gaussians: Number of different pairwise independence mixtures of the leaf nodes.

        """
        # Init
        super().__init__()
        self.n_gaussians = n_gaussians
        self.in_features = in_features

        self.gauss = Normal(multiplicity=3, in_channels=1, in_features=in_features)
        self.prod = layers.Product(in_features=128, cardinality=2, randomize=True)
        self.sum = layers.Sum(in_channels=3, in_features=64, out_channels=1)
        self.out = layers.Product(in_features=64, cardinality=64, randomize=True)

    def forward(self, x):
        x = self.gauss(x)
        x = self.prod(x)
        x = self.sum(x)
        x = self.out(x)
        return x


class SPNNeuronBig(nn.Module):
    """
    SPN Neuron implementation using the vectorized layers.
    """

    def __init__(self, in_features, n_gaussians=3):
        """
        Initialize the SPNNeuron.

        Args:
            in_features: Number of input features.
            n_gaussians: Number of different pairwise independence mixtures of the leaf nodes.

        """
        # Init
        super().__init__()
        self.n_gaussians = n_gaussians
        self.in_features = in_features

        ch = [2 ** i for i in range(6, 1, -1)]

        self.spn = nn.Sequential(
            Normal(multiplicity=n_gaussians, in_channels=1, in_features=in_features),
            layers.Sum(
                in_channels=n_gaussians, in_features=in_features, out_channels=ch[0]
            ),
            layers.Product(in_features=in_features, cardinality=2, randomize=True),
            layers.Sum(
                in_channels=ch[0], in_features=in_features / 2, out_channels=ch[1]
            ),
            layers.Product(in_features=in_features / 2, cardinality=2, randomize=True),
            layers.Sum(
                in_channels=ch[1], in_features=in_features / 4, out_channels=ch[2]
            ),
            layers.Product(in_features=in_features / 4, cardinality=2, randomize=True),
            layers.Sum(
                in_channels=ch[2], in_features=in_features / 8, out_channels=ch[3]
            ),
            layers.Product(in_features=in_features / 8, cardinality=2, randomize=True),
            layers.Sum(in_channels=ch[3], in_features=in_features / 16, out_channels=1),
            layers.Product(in_features=in_features / 16, cardinality=in_features // 16),
        )

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        x = self.spn(x)
        return x


class SPNNeuron(nn.Module):
    def __init__(self, in_features, n_gaussians=3):
        """
        Initialize the SPNNeuron.

        Args:
            in_features: Number of input features.
            n_gaussians: Number of different pairwise independence mixtures of the leaf nodes.

        """
        # Init
        super().__init__()
        self.n_gaussians = n_gaussians
        self.in_features = in_features

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(n_gaussians, 1, in_features))
        self.stds = nn.Parameter(torch.rand(n_gaussians, 1, in_features))
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

        # Create random sequence of scopes
        scopes = np.random.permutation(in_features)

        # sums = []

        self.product_scopes_left = []
        self.product_scopes_right = []

        # Sum weights
        self.sum_weights = nn.Parameter(torch.rand(n_gaussians, int(in_features / 2)))

        # For two consecutive (random) scopes
        for i in range(0, in_features, 2):
            # Collect scopes
            self.product_scopes_left.append(scopes[i])
            self.product_scopes_right.append(scopes[i + 1])

    def forward(self, x, marginals=[]):
        # First apply gaussian
        # Expand x to match the gaussian mean/std matrix
        batch_size = x.shape[0]
        x = x.expand([self.n_gaussians, batch_size, self.in_features])
        x = self.gauss.log_prob(x)

        # Marginalize certain leaf nodes: Set likelihood of the leaf to 1 (log(1)=0)
        x[:, :, marginals] = 0.0

        # Current dimensions: n_gaussian x batch_size x in_features
        #    ______
        #   /     /|
        #  /_____/ |
        #  |    |  /
        #  |    | /
        #  |____|/

        # Apply products between features i and j: X[:, :, i] * X[:, :, j] ( sum in logspace )
        x = x[:, :, self.product_scopes_left] + x[:, :, self.product_scopes_right]

        # Current dimensions: n_gaussian x batch_size x in_features / 2
        #    ______
        #   /     /|
        #  /_____/ |
        #  |    |  /
        #  |    | /
        #  |____|/

        # Apply sum over the n_gaussian axis (dim=0)
        x = torch.logsumexp(x + torch.log(self.sum_weights.unsqueeze(1)), dim=0)

        # Current dimensions: batch_size x in_features / 2
        #   ____
        #  |    |
        #  |    |
        #  |____|

        # Apply product over all features ( sum in logspace )
        x = torch.sum(x, dim=1)

        return x


class MaxOutSpnNeuron(nn.Module):
    def __init__(self, in_features, n_gaussians=3):
        """
        Initialize the SPNNeuron.

        Args:
            in_features: Number of input features.
            n_gaussians: Number of different pairwise independence mixtures of the leaf nodes.

        """
        # Init
        super(MaxOutSpnNeuron, self).__init__()
        self.n_gaussians = n_gaussians
        self.in_features = in_features

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(n_gaussians, 1, in_features))
        self.stds = nn.Parameter(torch.rand(n_gaussians, 1, in_features))
        self.gauss = dist.Normal(loc=self.means, scale=self.stds)

        # Create random sequence of scopes
        scopes = np.random.permutation(in_features)

        # sums = []

        self.product_scopes_left = []
        self.product_scopes_right = []

        # Sum weights
        self.sum_weights = nn.Parameter(torch.rand(n_gaussians, int(in_features / 2)))

        # For two consecutive (random) scopes
        for i in range(0, in_features, 2):
            # Collect scopes
            self.product_scopes_left.append(scopes[i])
            self.product_scopes_right.append(scopes[i + 1])

    def forward(self, x, marginals=[]):
        # First apply gaussian
        # Expand x to match the gaussian mean/std matrix
        batch_size = x.shape[0]
        x = x.expand([self.n_gaussians, batch_size, self.in_features])
        x = self.gauss.log_prob(x)

        # Marginalize certain leaf nodes: Set likelihood of the leaf to 1 (log(1)=0)
        x[:, :, marginals] = 0.0

        # Current dimensions: n_gaussian x batch_size x in_features
        #    ______
        #   /     /|
        #  /_____/ |
        #  |    |  /
        #  |    | /
        #  |____|/

        # Apply products between features i and j: X[:, :, i] * X[:, :, j] ( sum in logspace )
        x = x[:, :, self.product_scopes_left] + x[:, :, self.product_scopes_right]

        # Current dimensions: n_gaussian x batch_size x in_features / 2
        #    ______
        #   /     /|
        #  /_____/ |
        #  |    |  /
        #  |    | /
        #  |____|/

        # The above is similar to the maxout approch but instead returns a weighted sum for each scope
        x, _ = torch.max(x + torch.log(self.sum_weights.unsqueeze(1)), dim=0)
        # x, _ = torch.max(x, dim=0)

        # Current dimensions: batch_size x in_features / 2
        #   ____
        #  |    |
        #  |    |
        #  |____|

        # Apply product over all features ( sum in logspace )
        x = torch.sum(x, dim=1)

        return x


class ConditionalSPNNeuron(nn.Module):
    """
    Maps each input feature to the likeliood of that feature, given all other features:

        z_i = P(x_i | X \ {x_i})

    Dimension in: N, dimension out: N
    """

    def __init__(self, in_features: int):
        # Init
        super(ConditionalSPNNeuron, self).__init__()
        self.spn = SPNNeuron(in_features=in_features)
        self.in_features = in_features

    def forward(self, x):
        x_full_pass = self.spn(x)
        x_marginalized = [self.spn(x, i) for i in range(self.in_features)]
        x_stacked = torch.stack(x_marginalized, dim=1)
        x_conditional = x_full_pass.view(-1, 1) - x_stacked
        return x_conditional


class SPNNeuronOld(nn.Module):
    def __init__(self, in_features, n_mv=2):
        """
        Initialize the SPNNeuron.
        Args:
            in_features: Number of input features.
            n_mv: Number of different pairwise independence mixtures of the leaf nodes.
        """
        # Init
        super(SPNNeuronOld, self).__init__()

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


##########
# Layers #
##########


class SPNPosteriorLayer(nn.Module):
    """
    A PyTorch module that contains multiple SPNs with the same structure and treats them as single nodes in a layer.
    """

    def __init__(self, neuron: nn.Module, in_features: int, n_labels: int):
        """
        Initialize the SPNLayer.

        Args:
            in_features (int): Number of input features for this layer.
            n_labels (int): Number of output labels for this layer.
        """
        super().__init__()

        # Create out_features number of SPNNeurons
        neurons = [neuron(in_features) for _ in range(n_labels)]
        self.spns = nn.ModuleList(neurons)
        self.class_weights_log = nn.Parameter(
            torch.log(torch.ones(n_labels) / n_labels), requires_grad=False
        )
        self.n_labels = n_labels
        # self.class_distribution = nn.Parameter(torch.ones(n_labels) / n_labels)

    def forward(self, x):
        # Feed forward each neuron and stack the results
        spn_results = [spn(x) for spn in self.spns]
        x = torch.stack(spn_results, dim=1)
        x.squeeze_(2)

        # Normalize:   S(y=i | x) = S_i(x) * w_i / sum_i { w_i p_i(x) }
        # In logspace: S(y=i | x) = S_i(x) + log(w_i) - logsumexp{ log(w_i) + p_i(x) }
        # w_i = 1/n_labels
        z = torch.logsumexp(self.class_weights_log + x, dim=1).view(x.shape[0], 1)
        y = x + self.class_weights_log - z
        return y


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
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # Feed forward each neuron and stack the results
        spn_results = [spn(x) for spn in self.spns]
        x = torch.stack(spn_results, dim=1)
        x = x.squeeze()
        x = self.bn(x)
        return x


############
# Networks #
############


class ResNet(nn.Module):
    def __init__(self, in_features, n_labels, resnet_arch=resnet18):
        """
        Resnet.
        Args:
            in_features: Number of input features.
            n_labels: Number of output labels.
            resnet_arch: Resnet architecture.
        """
        super(ResNet, self).__init__()

        self.n_labels = n_labels
        self.resnet = resnet_arch(pretrained=False, num_classes=128, in_channels=1)
        self.linear = nn.Linear(128, n_labels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = self.linear(x)
        return x.sigmoid()


class SPNResnetParallel(nn.Module):
    def __init__(self, in_features, n_labels, resnet_arch):
        """
        Apply Resnet and SPN on the input in parallel, merge results and classify with 256 dimensional linear layer
        afterwards.

        Args:
            in_features: Number of input features.
            n_labels: Number of output labels.
            resnet_arch: Resnet architecture.
        """
        super(SPNResnetParallel, self).__init__()

        self.n_labels = n_labels
        self.resnet = resnet18(pretrained=False, num_classes=128, in_channels=1)
        self.mid = SPNLayer(neuron=SPNNeuron, in_features=in_features, out_features=128)
        self.linear = nn.Linear(128 * 2, n_labels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x_resnet = F.relu(self.resnet(x))
        x_spn = self.mid(x.view(x.shape[0], -1))
        x_concat = torch.cat([x_resnet, x_spn], dim=1)
        x = self.linear(x_concat)
        return x.sigmoid()


class SPNNetPure(nn.Module):
    def __init__(self, in_features, n_labels, spnneuron=SPNNeuron):
        """
        Apply SPN on input and directly produce the output.

        Args:
            in_features: Number of input features.
            n_labels: Number of output labels.
        """
        super(SPNNetPure, self).__init__()

        self.n_labels = n_labels
        self.mid = SPNLayer(
            neuron=SPNNeuronBig, in_features=in_features, out_features=n_labels
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.mid(x)
        return x.sigmoid()


class SPNPosteriorNet(nn.Module):
    def __init__(
        self, in_features, n_labels, resnet_arch=resnet18, spnneuron=SPNNeuron
    ):
        """
        Apply Resnet and SPN sequentially.

        SPN models the posterior distribution P(Y | X)

        Args:
            in_features: Number of input features.
            n_labels: Number of output labels.
            resnet_arch: Resnet architecture.
            spnneuron: SPN neuron type that defines the SPN architecture.
        """
        super().__init__()

        self.n_labels = n_labels
        self.resnet = resnet_arch(pretrained=False, num_classes=128, in_channels=3)
        self.mid = SPNPosteriorLayer(
            neuron=spnneuron, in_features=128, n_labels=n_labels
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = x.unsqueeze(2)
        x = self.mid(x)
        return x


class SPNNet(nn.Module):
    def __init__(
        self, in_features, n_labels, resnet_arch=resnet18, spnneuron=SPNNeuron
    ):
        """
        Apply Resnet and SPN sequentially.

        Args:
            in_features: Number of input features.
            n_labels: Number of output labels.
            resnet_arch: Resnet architecture.
            spnneuron: SPN neuron type that defines the SPN architecture.
        """
        super(SPNNet, self).__init__()

        self.n_labels = n_labels
        self.resnet = resnet_arch(pretrained=False, num_classes=128, in_channels=1)
        self.mid = SPNLayer(neuron=spnneuron, in_features=128, out_features=n_labels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = x.unsqueeze(2)
        x = self.mid(x)
        return x.sigmoid()


class ConditionalSPNNet(nn.Module):
    def __init__(self, in_features, n_labels, resnet_arch=resnet18):
        """
        Apply Resnet and then conditional SPN as defined by the `ConditionalSPNNeuron`.

        Args:
            in_features: Number of input features.
            n_labels: Number of output labels.
            resnet_arch: Resnet architecture.
        """
        super(ConditionalSPNNet, self).__init__()

        self.n_labels = n_labels
        self.resnet = resnet_arch(pretrained=False, num_classes=128, in_channels=1)
        self.mid = ConditionalSPNNeuron(128)
        self.out = nn.Linear(128, n_labels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = self.mid(x)
        x = self.out(x)
        return x.sigmoid()


def get_model_by_tag(tag, device, args, in_features, n_labels) -> nn.Module:
    """
    Return the model for a given tag.

    Args:
        tag (str): Model tag.
        device: Device to create the model on.
        args: Arguments

    Returns:
        nn.Module: PyTorch model.
    """

    resnet_arch = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }.get(args.resnet_arch)

    logger.info("Selecting model %s with device %s", tag, device)

    # Select model
    if tag.lower() == "resnet+spn":
        model = SPNNet(
            in_features=in_features, n_labels=n_labels, spnneuron=SPNNeuronVectorized
        ).to(device)
    elif tag.lower() == "resnet+posterior+spn":
        model = SPNPosteriorNet(
            in_features=in_features, n_labels=n_labels, spnneuron=SPNNeuronVectorized
        ).to(device)
    elif tag.lower() == "layerwise-spn-big":
        model = SPNNet(
            in_features=in_features, n_labels=n_labels, spnneuron=SPNNeuronBig
        ).to(device)
    elif tag.lower() == "pure-spn":
        model = SPNNetPure(in_features=in_features, n_labels=n_labels).to(device)
    elif tag.lower() == "resnet+spn-maxout":
        model = SPNNet(
            in_features=in_features,
            n_labels=n_labels,
            resnet_arch=resnet_arch,
            spnneuron=MaxOutSpnNeuron,
        ).to(device)
    elif tag.lower() == "parallel":
        model = SPNResnetParallel(
            in_features=in_features, n_labels=n_labels, resnet_arch=resnet_arch
        ).to(device)
    elif tag.lower() == "resnet":
        model = ResNet(
            in_features=in_features, n_labels=n_labels, resnet_arch=resnet_arch
        ).to(device)
    elif tag.lower() == "cond-spn":
        model = ConditionalSPNNet(
            in_features=in_features, n_labels=n_labels, resnet_arch=resnet_arch
        ).to(device)
    else:
        raise Exception("Invalid network: %s" % tag)

    # multi_gpu = len(args.cuda_device_id) > 1 or args.cuda_device_id[0] == -1

    # # Check if multiple cuda devices are selected
    # if multi_gpu:
    #     logger.info("Using multiple gpus")
    #     logger.info("args.cuda_device_id=%s", args.cuda_device_id)
    #     num_cuda_devices = torch.cuda.device_count()

    #     if args.cuda_device_id[0] == -1:
    #         # Select all devices
    #         cuda_device_id = list(range(num_cuda_devices))
    #     else:
    #         cuda_device_id = list(range(len(args.cuda_device_id)))

    #     # Check if multiple cuda devices are available
    #     if num_cuda_devices > 1:
    #         logger.info("Running experiment on the following GPUs: %s", cuda_device_id)

    #         # Transform model into data parallel model on all selected cuda deviecs
    #         model = torch.nn.DataParallel(model, device_ids=cuda_device_id)
    #     else:
    #         logger.warning(
    #             "Attempted to run the experiment on multiple GPUs while only %s GPU was available",
    #             num_cuda_devices,
    #         )

    return model
