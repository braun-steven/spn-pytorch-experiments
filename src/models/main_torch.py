import time

import sys
import traceback

sys.path.insert(0, "./")
from src.utils.args import parse_args
from torchsummary import summary
import argparse
import logging
import os
import sys
from typing import List
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
from torchvision import transforms

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

from src.utils.utils import ensure_dir, time_delta_now

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler("log.txt", mode="a"),
    ],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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
        self.fc1 = nn.Linear(in_features, 20)
        self.fc2 = nn.Linear(20, 5)
        self.fc3 = nn.Linear(5, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)


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
        self.fc1 = nn.Linear(in_features, 20)
        self.spn1 = SPNLayer(neuron=SPNNeuron, in_features=20, out_features=5)
        self.fc2 = nn.Linear(5, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.spn1(x)
        x = torch.log(-x)
        x = torch.max(x) - x
        x = self.fc2(x)
        return F.sigmoid(x)


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
        for g_a in gs0:
            for g_b in gs0:
                prods0.append(ProductNode(children=[g_a, g_b]))

        for g_a in gs1:
            for g_b in gs1:
                prods1.append(ProductNode(children=[g_a, g_b]))

        # Collect sum nodes
        sums = []
        n_sum_nodes = in_features ** 2
        weights = np.random.rand(n_sum_nodes, 2)
        weights = weights / weights.sum(axis=1, keepdims=True)
        for i in range(n_sum_nodes):
            sum_node = SumNode(
                weights=weights[i, :], children=[prods0[i], prods1[i]]
            )  # , prods2[i]])
            sums.append(sum_node)

        # Construct root node as product of all sums
        self.root = ProductNode(children=sums)

    def forward(self, x):
        return self.root(x)


def train(model, device, train_loader, optimizer, epoch):
    """
    Train the given model for one epoch.

    Args:
        model: PyTorch module.
        device: Execution device.
        train_loader: Training data loader.
        optimizer: Training weight optimizer.
        epoch (int): Current epoch.
    """
    model.train()
    clipper = SPNClipper(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
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


def evaluate_model(model: nn.Module, device, loader, tag) -> float:
    """
    Description for method evaluate_model.

    Args:
        model (nn.Module): PyTorch module.
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.binary_cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output > 0.5
            correct += pred.long().eq(target.long().view_as(pred)).sum().item()

    loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    logger.info(
        "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            tag, loss, correct, len(loader.dataset), accuracy
        )
    )
    return (loss, accuracy)


def main_datasets(dataset_name, X, y):
    """
    Main method to run an experiment on a given dataset.

    Args:
        dataset_name (str): Dataset name.
        X (np.ndarray): Input features.
        y (np.ndarray): Input labels.
    """
    # torch setup
    use_cuda = ARGS.cuda and torch.cuda.is_available()
    torch.manual_seed(ARGS.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Select model
    if ARGS.net == "spn":
        model = SPNNet(X.shape[1]).to(device)
    elif ARGS.net == "mlp":
        model = MLPNet(X.shape[1]).to(device)
    else:
        raise Exception("Invalid network: %s" % ARGS.net)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)

    def get_dataloader(X, y, batch_size):
        tensors = [torch.tensor(d, dtype=torch.float32) for d in [X, y]]
        ds = torch.utils.data.TensorDataset(*tensors)
        # Prepare training
        return torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size)

    train_loader = get_dataloader(X_train, y_train, ARGS.batch_size)
    test_loader = get_dataloader(X_test, y_test, ARGS.test_batch_size)

    # Halven the learning rate after 25 epochs
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
        result_dir=os.path.join(ARGS.result_dir, ARGS.net),
        dataset_name=dataset_name,
        column_names=column_names,
        data=data,
    )


def main_run_experiment(exp_method):
    """
    Run a main experiment. Dataset is chosen by ARGS.dataset flag.

    Args:
        exp_method (funtion): Experiment function, which is being called with (datasetname, X, y).
    """
    dss = data_loader.load_dataset_map()
    name = ARGS.dataset
    loader = dss[name]
    X, y = loader()

    # If debug: shuffle X, y and only take the first 20 elements
    if ARGS.debug:
        X, y = shuffle(X, y)
        X, y = X[:20], y[:20]
    logger.info(" -- %s -- Shape: %s", name, X.shape)
    t = time.time()
    exp_method(dataset_name=name, X=X, y=y)
    tstr = time_delta_now(t)
    logger.info(" -- %s -- Finished, took %s", name, tstr)


if __name__ == "__main__":
    ARGS = parse_args()
    main_run_experiment(main_datasets)
