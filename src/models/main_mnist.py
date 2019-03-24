import time


import multiprocessing
from joblib import Parallel, delayed
from src.utils.args import parse_args
import argparse
import logging
import os
import sys
from typing import List, Tuple
from src.data.data import store_results
from src.data.data_loader import get_mnist_loaders

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import distributions as dist
from torch import nn
from torchvision import transforms, datasets

from src.data import data_loader
from src.models.pytorch import (
    CategoricalNode,
    GaussianNode,
    ProductNode,
    SPNClipper,
    SPNLayer,
    SumNode,
    optimize_torch,
)
from spn.structure.Base import Node, Product, Sum
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

from src.utils.utils import (
    setup_logging,
    ensure_dir,
    time_delta_now,
    count_params,
    generate_run_base_dir,
)

# Logger instance
logger = logging.getLogger(__name__)


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
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.spn1(x)
        x = torch.log(-x)
        x = 1 / x  # TODO: What about zeros?
        # x = self.bn1(x)  # TODO: Check out effect of batch norm
        x = self.fc3(x)
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


def main_mnist():
    """Run the MNIST experiment."""
    use_cuda = not ARGS.no_cuda and torch.cuda.is_available()
    torch.manual_seed(ARGS.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get the mnist loader
    train_loader, test_loader = get_mnist_loaders(use_cuda, ARGS)

    # Select model
    if ARGS.net == "spn":
        model = SPNNet(in_features=28 * 28).to(device)
    elif ARGS.net == "mlp":
        model = MLPNet(in_features=28 * 28).to(device)
    else:
        raise Exception("Invalid network: %s" % ARGS.net)

    # with SummaryWriter(comment="Model", log_dir="tensorboard") as w:
    #     w.add_graph(model, torch.zeros(1, 28, 28), True)
    # exit()

    logger.info("Number of paramters: %s", count_params(model))
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)

    # Scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # Collect acc and loss
    train_accs, test_accs = [], []
    train_losses, test_losses = [], []

    # Run epochs
    for epoch in range(1, ARGS.epochs + 1):
        scheduler.step()
        # Run train
        train(model, device, train_loader, optimizer, epoch)

        # Evaluate model on train and test data
        train_loss, train_acc = evaluate_model(model, device, train_loader, "Train")
        test_loss, test_acc = evaluate_model(model, device, test_loader, "Test")

        # Store acc/loss
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

    # Store results
    column_names = ["train_acc", "test_acc", "train_loss", "test_loss"]
    data = np.c_[train_accs, test_accs, train_losses, test_losses]
    store_results(
        result_dir=base_dir,
        dataset_name="mnist",
        exp_name=ARGS.net,
        column_names=column_names,
        data=data,
    )


def train(model, device, train_loader, optimizer, epoch):
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
    clipper = SPNClipper()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.apply(clipper)
        if batch_idx % ARGS.log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def evaluate_model(
    model: nn.Module, device: str, loader, tag: str
) -> Tuple[float, float]:
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): Network model.
        device: Torch device to evaluate on.
        loader: Torch dataset loader.
        tag: Tag for logging (Train/Test).

    Returns:
        Tuple[float, float]: Tuple of (loss, accuracy).
    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.long().eq(target.long().view_as(pred)).sum().item()

    loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    logger.info(
        "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            tag, loss, correct, len(loader.dataset), accuracy
        )
    )
    return (loss, accuracy)


if __name__ == "__main__":
    ARGS = parse_args()
    base_dir = generate_run_base_dir(ARGS.tag, ARGS.result_dir)
    log_file = os.path.join(base_dir, "log.txt")

    # Setup logging in base_dir/log.txt
    setup_logging(level=ARGS.log_level, filename=log_file)
    logger.info(" -- MNIST -- ")
    t = time.time()
    main_mnist()
    tstr = time_delta_now(t)
    logger.info(" -- MNIST -- Finished, took %s", tstr)
