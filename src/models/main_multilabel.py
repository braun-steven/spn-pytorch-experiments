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

from src.data.data import store_results
from src.data.data_loader import get_mnist_subset, get_multilabel_mnist_loader
from src.models.experiment import MnistExperiment
from src.models.mnist import get_n_samples_from_loader
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

###############################
# 128 > 64 > 32 > 10*n_labels #
###############################

# class MLPNet(nn.Module):
#     def __init__(self, in_features, n_labels):
#         super(MLPNet, self).__init__()
#         self.n_labels = n_labels
#         self.fc1 = nn.Linear(in_features, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 10 * n_labels)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.kaiming_normal_(m.weight)

#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         x = x.view(-1, 10, self.n_labels)
#         # return F.sigmoid(x)
#         return F.log_softmax(x, dim=1)


# class SPNNet(nn.Module):
#     def __init__(self, in_features, n_labels):
#         super(SPNNet, self).__init__()
#         self.n_labels = n_labels

#         # Define Layers
#         self.fc1 = nn.Linear(in_features, 128)
#         self.spn1 = SPNLayer(neuron=SPNNeuron, in_features=128, out_features=64)
#         self.spn2 = SPNLayer(neuron=SPNNeuron, in_features=64, out_features=32)
#         self.fc2 = nn.Linear(32, 10 * n_labels)

#         # Init weights
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.kaiming_normal_(m.weight)

#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # Reshape height and width into single dimension
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))  # First linear layer
#         x = self.spn1(x)  # SPN
#         x = self.spn2(x)  # SPN
#         x = self.fc2(x)  # Linear
#         x = x.view(-1, 10, self.n_labels)
#         return F.log_softmax(x, dim=1)

#########################
# 32 > 20 > 10*n_labels #
#########################


class MLPNet(nn.Module):
    def __init__(self, in_features, n_labels):
        super(MLPNet, self).__init__()
        self.n_labels = n_labels
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 20)
        self.fc3 = nn.Linear(20, n_labels * 10)
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
        x = x.view(-1, 10, self.n_labels)
        # return F.sigmoid(x)
        return F.log_softmax(x, dim=1)


class SPNNet(nn.Module):
    def __init__(self, in_features, n_labels):
        super(SPNNet, self).__init__()
        self.n_labels = n_labels

        # Define Layers
        self.fc1 = nn.Linear(in_features, 32)
        self.spn1 = SPNLayer(neuron=SPNNeuron, in_features=32, out_features=20)
        self.fc2 = nn.Linear(20, 10 * n_labels)

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
        x = x.view(-1, 10, self.n_labels)
        return F.log_softmax(x, dim=1)


#################################

# class MLPNet(nn.Module):
#     def __init__(self, in_features, n_labels):
#         super(MLPNet, self).__init__()
#         self.n_labels = n_labels
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
#         self.fc1 = nn.Linear(in_features, 32)
#         self.fc2 = nn.Linear(32, 20)
#         self.fc3 = nn.Linear(20, n_labels * 10)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.kaiming_normal_(m.weight)

#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = x.view(-1, 10, self.n_labels)
#         # return F.sigmoid(x)
#         return F.log_softmax(x, dim=1)


# class SPNNet(nn.Module):
#     def __init__(self, in_features, n_labels):
#         super(SPNNet, self).__init__()
#         self.n_labels = n_labels

#         # Define Layers
#         self.fc1 = nn.Linear(in_features, 32)
#         self.spn1 = SPNLayer(neuron=SPNNeuron, in_features=32, out_features=20)
#         self.fc2 = nn.Linear(20, 10 * n_labels)

#         # Init weights
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.kaiming_normal_(m.weight)

#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # Reshape height and width into single dimension
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))  # First linear layer
#         x = self.spn1(x)  # SPN
#         x = self.fc2(x)  # Linear
#         x = x.view(-1, 10, self.n_labels)
#         return F.log_softmax(x, dim=1)


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
        model = SPNNet(in_features=28 * 28 * ARGS.n_labels, n_labels=ARGS.n_labels).to(
            device
        )
    elif tag.lower() == "mlp":
        model = MLPNet(in_features=28 * 28 * ARGS.n_labels, n_labels=ARGS.n_labels).to(
            device
        )
    else:
        raise Exception("Invalid network: %s" % tag)

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
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target_oh = make_one_hot(target.unsqueeze(1), C=10)
            loss += F.binary_cross_entropy_with_logits(
                output, target_oh, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.long().eq(target.long().view_as(pred)).sum().item()

    n_samples = get_n_samples_from_loader(loader) * n_labels

    loss /= n_samples
    accuracy = 100.0 * correct / n_samples

    logger.info(
        "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            tag, loss, correct, n_samples, accuracy
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

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = make_one_hot(target.unsqueeze(1), C=10)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        model.apply(clipper)
        if batch_idx % log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    n_samples,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def run_multilabel_mnist(args):
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
    device = torch.device("cuda" if use_cuda else "cpu")
    bs = ARGS.batch_size

    # Get the mnist loader
    train_loader, test_loader = get_multilabel_mnist_loader(
        n_labels=ARGS.n_labels, use_cuda=use_cuda, args=args
    )
    # Retreive model
    model = get_model_by_tag(args.net, device)

    logger.info("Number of paramters: %s", count_params(model))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    data = []
    # Run epochs
    for epoch in range(1, args.epochs + 1):
        # Start counting after 10 epochs, that is, first lr reduction is at epoch 20
        if epoch > 10:
            scheduler.step()

        # Run train
        train_multilabel(
            model, device, train_loader, optimizer, epoch, args.log_interval
        )

        # Evaluate model on train/test data
        train_loss, train_acc = evaluate_model_multilabel(
            model, device, train_loader, "Train", ARGS.n_labels
        )
        test_loss, test_acc = evaluate_model_multilabel(
            model, device, test_loader, "Test", ARGS.n_labels
        )
        data.append([epoch, train_acc, test_acc, train_loss, test_loss])

    column_names = ["epoch", "train_acc", "test_acc", "train_loss", "test_loss"]
    store_results(
        result_dir=os.path.join(args.result_dir, args.experiment_name),
        dataset_name="mnist",
        column_names=column_names,
        data=data,
    )

    # Evaluate model on train/test data
    train_loss, train_acc = evaluate_model_multilabel(
        model, device, train_loader, "Train", ARGS.n_labels
    )
    test_loss, test_acc = evaluate_model_multilabel(
        model, device, test_loader, "Test", ARGS.n_labels
    )


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
        run_multilabel_mnist(ARGS)
    except Exception as e:
        logger.exception("Experiment crashed.")
        logger.exception("Exception: %s", str(e))

    # Measure time
    tstr = time_delta_now(tstart)
    logger.info(" -- MNIST -- Finished, took %s", tstr)


if __name__ == "__main__":
    ARGS = parse_args()
    main()
