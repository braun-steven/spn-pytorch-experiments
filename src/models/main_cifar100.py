import logging
import os
import sys
import time
import traceback
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import distributions as dist

from src.data.data import store_results
from src.data.data_loader import get_cifar100_loader
from src.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from src.models.mnist import get_n_samples_from_loader
from src.models.pytorch import (
    GaussianNode,
    MultivariateGaussian,
    ProductNode,
    SPNClipper,
    SPNLayer,
    SumNode,
)
from src.utils.args import init_argparser
from src.utils.args import save_args
from src.utils.utils import (
    count_params,
    ensure_dir,
    make_one_hot,
    set_seed,
    setup_logging,
    time_delta_now,
)

os.environ["OMP_NUM_THREADS"] = "1"


sys.path.insert(0, "./")


logger = logging.getLogger(__name__)


def parse_args():
    parser = init_argparser()

    # Specific arguments for this experiment

    args = parser.parse_args()
    return args


class ResNet(nn.Module):
    def __init__(self, in_features, n_labels, resnet_arch=resnet18):
        super(ResNet, self).__init__()

        self.n_labels = n_labels
        self.resnet = resnet_arch(pretrained=False, num_classes=128)
        self.mid = nn.Linear(128, n_labels)
        self.bn = nn.BatchNorm1d(n_labels)
        # self.out = nn.Linear(64, n_labels)
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
        x = F.relu(self.mid(x))
        x = self.bn(x)
        return x.sigmoid()


class SPNResnetParallel(nn.Module):
    def __init__(self, in_features, n_labels, resnet_arch):
        super(SPNResnetParallel, self).__init__()

        self.n_labels = n_labels
        self.resnet = resnet_arch(pretrained=False, num_classes=128, in_channel=1)
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


class SPNNet(nn.Module):
    def __init__(self, in_features, n_labels, resnet_arch=resnet18):
        super(SPNNet, self).__init__()

        self.n_labels = n_labels
        self.resnet = resnet_arch(pretrained=False, num_classes=128)
        self.mid = SPNLayer(neuron=SPNNeuron, in_features=128, out_features=n_labels)
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
        return x.sigmoid()


class ConditionalSPNNet(nn.Module):
    def __init__(self, in_features, n_labels, resnet_arch=resnet18):
        super(ConditionalSPNNet, self).__init__()

        self.n_labels = n_labels
        self.resnet = resnet_arch(pretrained=False, num_classes=128)
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


class SPNNeuron(nn.Module):
    def __init__(self, in_features, n_gaussians=3):
        """
        Initialize the SPNNeuron.

        Args:
            in_features: Number of input features.
            n_gaussians: Number of different pairwise independence mixtures of the leaf nodes.

        """
        # Init
        super(SPNNeuron, self).__init__()
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


def get_model_by_tag(tag, device, args) -> nn.Module:
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

    # Select model
    if tag.lower() == "resnet+spn":
        model = SPNNet(
            in_features=32 * 32 * 3, n_labels=100, resnet_arch=resnet_arch
        ).to(device)
    elif tag.lower() == "parallel":
        model = SPNResnetParallel(
            in_features=32 * 32 * 3, n_labels=100, resnet_arch=resnet_arch
        ).to(device)
    elif tag.lower() == "resnet":
        model = ResNet(
            in_features=32 * 32 * 3, n_labels=100, resnet_arch=resnet_arch
        ).to(device)
    elif tag.lower() == "cond-spn":
        model = ConditionalSPNNet(
            in_features=32 * 32 * 3, n_labels=100, resnet_arch=resnet_arch
        ).to(device)
    else:
        raise Exception("Invalid network: %s" % tag)

    multi_gpu = len(args.cuda_device_id) > 1 or args.cuda_device_id[0] == -1

    # Check if multiple cuda devices are selected
    if multi_gpu:
        num_cuda_devices = torch.cuda.device_count()

        if args.cuda_device_id[0] == -1:
            # Select all devices
            cuda_device_id = list(range(num_cuda_devices))
        else:
            cuda_device_id = args.cuda_device_id

        # Check if multiple cuda devices are available
        if num_cuda_devices > 1:
            logger.info("Running experiment on the following GPUs: %s", cuda_device_id)

            # Transform model into data parallel model on all selected cuda deviecs
            model = torch.nn.DataParallel(model, device_ids=cuda_device_id)
        else:
            logger.warning(
                "Attempted to run the experiment on multiple GPUs while only %s GPU was available",
                num_cuda_devices,
            )

    return model


def evaluate_model_multilabel(
    model: nn.Module, device: str, loader, tag: str, n_labels: int
) -> Tuple[float, float]:
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): Network model.
        device: Torch device to evaluate on.
        loader: Torch dataset loader.
        tag: Tag for logging (Train/Test).
        n_labels: Number of labels.

    Returns:
        Tuple[float, float]: Tuple of (loss, accuracy).
    """
    model.eval()

    loss = 0.0
    correct = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for data, target in loader:
            # Send data and target to correct device
            data, target = data.to(device), target.to(device)

            # Do inference
            output = model(data)

            # Comput loss
            loss += loss_fn(output, target)

            _, pred = output.max(1)
            correct += pred.long().eq(target.long().view_as(pred)).sum().item()

    n_samples = get_n_samples_from_loader(loader) * n_labels

    loss /= n_samples
    accuracy = 100.0 * correct / n_samples

    logger.info(
        "{: <5} set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(
            tag, loss, accuracy
        )
    )
    return (loss, accuracy)


def train_multilabel(model, device, train_loader, optimizer, epoch, log_interval=10):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Network model.
        device: Device to train on.
        train_loader: Torch data loader for training set.
        optimizer: Torch opitimizer.
        epoch: Current epoch.
    """

    model.train()
    clipper = SPNClipper(device)

    n_samples = get_n_samples_from_loader(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    t_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Send data to correct device
        data, target = data.to(device), target.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Inference
        output = model(data)
        output = torch.softmax(output, dim=1)

        C = 100
        target_oh = torch.FloatTensor(target.size(0), C).zero_()
        target_oh[range(target.size(0)), target] = 1.0

        # Comput loss
        loss = loss_fn(output, target)

        # Backprop
        loss.backward()
        optimizer.step()

        # Clipp weights
        model.apply(clipper)

        # Log stuff
        if batch_idx % log_interval == 0:
            logger.info(
                "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    n_samples,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    t_delta = time_delta_now(t_start)
    logger.info("Train Epoch: {} took {}".format(epoch, t_delta))


def run_multilabel_cifar100(args):
    """
    Run the experiment with a given percentage.

    Args:
        percentage (float): Percentage of training data available.
        args: Command line args.

    Returns:
        Tuple[float, float, float, float]: Train acc, Test acc, Train loss, Test loss.
    """
    use_cuda = args.cuda and torch.cuda.is_available()
    # Set seed globally
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    cuda_device = "cuda:{}".format(args.cuda_device_id[0])
    device = torch.device(cuda_device if use_cuda else "cpu")

    logger.info("Main device: %s", device)
    bs = args.batch_size

    # Get the cifar100 loader
    train_loader, test_loader = get_cifar100_loader(use_cuda=use_cuda, args=args)

    # Retreive model
    model = get_model_by_tag(args.net, device, args)

    # Disable track_running_stats in batchnorm according to
    # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/12
    for child in model.modules():
        if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
            child.track_running_stats = False

    logger.info("Number of paramters: %s", count_params(model))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler for learning rate
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

    data = []
    # Run epochs
    for epoch in range(1, args.epochs + 1):
        # Start counting after 10 epochs, that is, first lr reduction is at epoch 20
        if epoch > 10:
            logger.info(
                "Epoch %s: Reducing learning rate by %s to %s",
                epoch,
                gamma,
                optimizer.param_groups[0]["lr"],
            )
            scheduler.step()

        # Run train
        train_multilabel(
            model, device, train_loader, optimizer, epoch, args.log_interval
        )

        # Evaluate model on train/test data
        train_loss, train_acc = evaluate_model_multilabel(
            model, device, train_loader, "Train", 100
        )
        test_loss, test_acc = evaluate_model_multilabel(
            model, device, test_loader, "Test", 100
        )
        data.append([epoch, train_acc, test_acc, train_loss, test_loss])

    column_names = ["epoch", "train_acc", "test_acc", "train_loss", "test_loss"]
    store_results(
        result_dir=os.path.join(args.result_dir, args.experiment_name),
        dataset_name="cifar100",
        column_names=column_names,
        data=data,
    )


def main():
    """Run the Cifar100 experiment."""
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={"float_kind": float_formatter})

    # Setup logging in base_dir/log.txt
    log_file = os.path.join(ARGS.result_dir, ARGS.experiment_name, "log.txt")
    setup_logging(level=ARGS.log_level, filename=log_file)
    logger.info(" -- Cifar100 -- Started ")
    print("Result dir: ", ARGS.result_dir)
    print("Log file: ", log_file)

    # Save commandline arguments
    save_args(ARGS)

    tstart = time.time()
    try:
        if not ARGS.cuda:
            # Set number of CPU threads
            torch.set_num_threads(ARGS.njobs)

        # Create and run experiment
        run_multilabel_cifar100(ARGS)
    except Exception as e:
        logger.exception("Experiment crashed.")
        logger.exception("Exception: %s", str(e))

    # Measure time
    tstr = time_delta_now(tstart)
    logger.info(" -- CIFAR100 -- Finished, took %s", tstr)


def plot_sample(x, y, y_pred, loss):
    """
    Plot a single sample witht the target and prediction in the title.

    Args:
        x: Image.
        y: Target.
        y_pred: Target prediction.
        loss: Loss value.
    """
    import matplotlib.pyplot as plt

    plt.imshow(x.squeeze().numpy())
    plt.title(
        "y={}\ny_pred={}\nloss={}".format(
            y.squeeze().numpy(),
            y_pred.squeeze().detach().numpy(),
            loss.detach().numpy(),
        )
    )
    plt.show()


if __name__ == "__main__":
    ARGS = parse_args()
    main()
