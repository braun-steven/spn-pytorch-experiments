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
from src.data.data_loader import (
    get_mnist_subset,
    get_multilabel_mnist_loader,
    load_multi_mnist,
)
from src.models.mnist import get_n_samples_from_loader
from src.models.pytorch import SPNClipper, SPNLayer
from src.utils.args import init_argparser
from src.utils.args import save_args
from src.utils.utils import (
    count_params,
    ensure_dir,
    set_seed,
    setup_logging,
    time_delta_now,
)
from src.models.models import *

os.environ["OMP_NUM_THREADS"] = "1"


sys.path.insert(0, "./")


logger = logging.getLogger(__name__)


def parse_args():
    parser = init_argparser()

    # Specific arguments for this experiment
    parser.add_argument(
        "--n-labels",
        type=int,
        default=2,
        metavar="N",
        help="Number of labels for artificial multilabel mnist task. Digits will be sample from [0, 1, ..., n_labels]",
    )
    parser.add_argument(
        "--canvas-size", type=int, default=64, metavar="N", help="Canvas size."
    )
    parser.add_argument(
        "--n-digits",
        type=int,
        default=5,
        metavar="N",
        help="Number of maximum digits per canvas.",
    )
    args = parser.parse_args()

    if args.n_digits > args.n_labels:
        raise Exception("Option --n-digits has to be <= --n-labels.")
    return args


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

    loss_fn = nn.BCELoss(reduction="sum")

    # Collect targets and outputs
    with torch.no_grad():
        for data, target in loader:
            # Send data and target to correct device
            data, target = data.to(device), target.to(device)

            # Do inference
            output = model(data)

            # Comput loss
            loss += loss_fn(output, target)

            pred = output > 0.5
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
    loss_fn = nn.BCELoss()
    t_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Send data to correct device
        data, target = data.to(device), target.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Inference
        output = model(data)

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
        # if batch_idx % log_interval * 3 == 0:
        #     logger.info("Samples:")
        #     logger.info("Target: %s", target[0].cpu().numpy())
        #     logger.info("Output: %s", output[0].detach().cpu().numpy())
    t_delta = time_delta_now(t_start)
    logger.info("Train Epoch: {} took {}".format(epoch, t_delta))


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
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logger.info("Main device: %s", device)

    # Get the mnist loader
    train_loader, test_loader = load_multi_mnist(
        n_labels=args.n_labels, canvas_size=args.canvas_size, seed=args.seed, args=args
    )

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


def main():
    """Run the MNIST experiment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in ARGS.cuda_device_id])
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={"float_kind": float_formatter})

    # Setup logging in base_dir/log.txt
    log_file = os.path.join(ARGS.result_dir, ARGS.experiment_name, "log.txt")
    setup_logging(level=ARGS.log_level, filename=log_file)
    logger.info(" -- MNIST Multilabel -- Started ")
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
        run_multilabel_mnist(ARGS)
    except Exception as e:
        logger.exception("Experiment crashed.")
        logger.exception("Exception: %s", str(e))

    # Measure time
    tstr = time_delta_now(tstart)
    logger.info(" -- MNIST -- Finished, took %s", tstr)


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
