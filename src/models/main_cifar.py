import logging
import os
import sys
import time
import traceback
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.nn import functional as F
from torch import distributions as dist

from src.spn.clipper import DistributionClipper, SumWeightNormalizer, SumWeightClipper

from src.data.data import store_results
from src.data.data_loader import get_cifar_loader
from src.utils.utils import get_n_samples_from_loader
from src.utils.utils import set_cuda_device
from src.models.models import get_model_by_tag
from src.utils.args import init_argparser
from src.utils.args import save_args
from src.utils.utils import (
    collect_tensorboard_info,
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
    parser.add_argument(
        "--cifar",
        type=int,
        default=10,
        metavar="N",
        help="Cifar dataset. Must be one of [10, 100]",
        choices=[10, 100],
    )

    args = parser.parse_args()
    return args


def evaluate(model: nn.Module, device: str, loader, tag: str) -> Tuple[float, float]:
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

    loss = 0.0
    correct = 0
    loss_fn = nn.NLLLoss(reduction="sum")

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

    n_samples = get_n_samples_from_loader(loader)

    loss /= n_samples
    accuracy = 100.0 * correct / n_samples

    logger.info(
        "{: <5} set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(
            tag, loss, accuracy
        )
    )
    return (loss, accuracy)


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
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
    dist_clipper = DistributionClipper(device)
    sum_weight_normalizer = SumWeightNormalizer()
    sum_weight_clipper = SumWeightClipper(device)

    n_samples = get_n_samples_from_loader(train_loader)
    loss_fn = nn.NLLLoss()
    t_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Send data to correct device
        data, target = data.to(device), target.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Inference
        output = model(data)
        loss = loss_fn(output, target)

        # Backprop
        loss.backward()
        optimizer.step()

        # Clip distribution values and weights
        model.apply(dist_clipper)
        model.apply(sum_weight_clipper)

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
    model.apply(sum_weight_normalizer)
    t_delta = time_delta_now(t_start)
    logger.info("Train Epoch: {} took {}".format(epoch, t_delta))


def run_cifar(args):
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

    # Get the cifar loader
    train_loader, test_loader = get_cifar_loader(
        n_labels=args.cifar, use_cuda=use_cuda, args=args
    )

    # Retreive model
    model = get_model_by_tag(
        in_features=32 * 32, tag=args.net, device=device, args=args, n_labels=args.cifar
    )

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

    writer = SummaryWriter(log_dir=os.path.join(ARGS.result_dir, ARGS.experiment_name))

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
        train(model, device, train_loader, optimizer, epoch, args.log_interval)

        # Evaluate model on train/test data
        train_loss, train_acc = evaluate(model, device, train_loader, "Train")
        test_loss, test_acc = evaluate(model, device, test_loader, "Test")
        data.append([epoch, train_acc, test_acc, train_loss, test_loss])

        # Collect data
        collect_tensorboard_info(
            writer, model, epoch, train_acc, test_acc, train_loss, test_loss
        )

    column_names = ["epoch", "train_acc", "test_acc", "train_loss", "test_loss"]
    store_results(
        result_dir=os.path.join(args.result_dir, args.experiment_name),
        dataset_name="cifar%s" % args.cifar,
        column_names=column_names,
        data=data,
    )


def main():
    """Run the Cifar experiment."""
    set_cuda_device(ARGS.cuda_device_id)
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={"float_kind": float_formatter})

    # Setup logging in base_dir/log.txt
    log_file = os.path.join(ARGS.result_dir, ARGS.experiment_name, "log.txt")
    setup_logging(level=ARGS.log_level, filename=log_file)
    logger.info(" -- Cifar -- Started ")
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
        run_cifar(ARGS)
    except Exception as e:
        logger.exception("Experiment crashed.")
        logger.exception("Exception: %s", str(e))

    # Measure time
    tstr = time_delta_now(tstart)
    logger.info(" -- CIFAR -- Finished, took %s", tstr)


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
