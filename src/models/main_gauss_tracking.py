import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import SubsetRandomSampler

from tensorboardX import SummaryWriter

from src.data.data import store_results
from src.data.data_loader import get_mnist_subset, get_multilabel_mnist_loader
from src.data.data_loader import get_mnist_loaders
from src.models.experiment import MnistExperiment
from src.models.mnist import get_n_samples_from_loader
from src.models.mnist import train
from src.models.mnist import evaluate_model
from src.models.pytorch import (
    GaussianNode,
    MultivariateGaussian,
    ProductNode,
    SPNClipper,
    SPNLayer,
    SumNode,
)
from src.utils.args import parse_args
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


class MLPNet(nn.Module):
    def __init__(self, in_features):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 20)
        self.fc3 = nn.Linear(20, 10)
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
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class SPNNet(nn.Module):
    def __init__(self, in_features):
        super(SPNNet, self).__init__()

        # Define Layers
        self.fc1 = nn.Linear(in_features, 32)
        self.spn1 = SPNLayer(neuron=SPNNeuron, in_features=32, out_features=20)
        self.fc2 = nn.Linear(20, 10)

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


def run_mnist(args):
    """
    Run the experiment.

    Args:
        args: Command line args.
    """
    use_cuda = args.cuda and torch.cuda.is_available()
    # Set seed globally
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get the mnist loader
    train_loader, test_loader = get_mnist_loaders(use_cuda=use_cuda, args=args)
    # Retreive model
    model = get_model_by_tag(args.net, device)

    logger.info("Number of paramters: %s", count_params(model))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    writer = SummaryWriter(
        log_dir=os.path.join(args.result_dir, args.experiment_name, "tensorboard")
    )

    # Run epochs
    for epoch in range(1, args.epochs + 1):
        # Stop after 2 epochs in debug mode
        if args.debug and epoch > 5:
            break

        # Start counting after 10 epochs, that is, first lr reduction is at epoch 20
        if epoch > 10:
            scheduler.step()

        for name, param in model.named_parameters():
            # Only check for first and second spn
            if "spn1.spns.0" in name or "spn1.spns.1" in name:
                if "mean" in name or "std" in name:
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

                # Track weights (use only first weight as this makes it easier to visualize and the sums only have two children for now anyway)
                if "weights" in name:
                    writer.add_histogram(
                        name, param.clone().cpu().data.numpy()[0], epoch
                    )

        # Run train
        train(model, device, train_loader, optimizer, epoch, args.log_interval)

        # Evaluate model on train/test data
        train_loss, train_acc = evaluate_model(model, device, train_loader, "Train")
        test_loss, test_acc = evaluate_model(model, device, test_loader, "Test")
        writer.add_scalar(
            tag="accuracy/train", scalar_value=train_acc, global_step=epoch
        )
        writer.add_scalar(tag="accuracy/test", scalar_value=test_acc, global_step=epoch)
        writer.add_scalar(tag="loss/train", scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag="loss/test", scalar_value=test_loss, global_step=epoch)

    writer.close()


def main():
    """Run the MNIST experiment."""
    log_file = os.path.join(ARGS.result_dir, ARGS.experiment_name, "log.txt")
    print("Result dir: %s", ARGS.result_dir)
    print("Log file: %s", log_file)

    # Setup logging in base_dir/log.txt
    setup_logging(level=ARGS.log_level, filename=log_file)
    logger.info(" -- MNIST Multilabel -- Started ")
    tstart = time.time()

    try:
        if not ARGS.cuda:
            # Set number of CPU threads
            torch.set_num_threads(ARGS.njobs)

        # Create and run experiment
        run_mnist(ARGS)
    except Exception as e:
        logger.exception("Experiment crashed.")
        logger.exception("Exception: %s", str(e))

    # Measure time
    tstr = time_delta_now(tstart)
    logger.info(" -- MNIST -- Finished, took %s", tstr)


if __name__ == "__main__":
    ARGS = parse_args()
    main()
