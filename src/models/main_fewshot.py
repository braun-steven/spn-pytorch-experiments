import sys
import traceback
import os

os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing as mp

sys.path.insert(0, "./")
import logging
import os
import time
import torch

from src.utils.args import parse_args
from src.utils.utils import ensure_dir, setup_logging, time_delta_now
from src.utils.utils import set_seed
from src.models.experiment import MnistExperiment
import logging

import numpy as np
import torch
from torch import optim
from typing import List
from src.utils.utils import count_params

from src.data.data import store_results
from src.data.data_loader import get_mnist_loaders
from src.data.data_loader import get_mnist_subset
from src.models.mnist import evaluate_model, train
from src.models.models import get_model_by_tag
from torch.utils.data import SubsetRandomSampler
from typing import Tuple

logger = logging.getLogger(__name__)


def run_for_percentage(percentage: float, args) -> Tuple[float, float, float, float]:
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
    bs = int(60000 * percentage / 100 * 1 / 10)
    logger.info("Current percentage: %.2f, Batch size: %s", percentage, bs)

    # Get the mnist loader
    train_loader, test_loader = get_mnist_subset(
        use_cuda=use_cuda, train_bs=bs, test_bs=args.test_batch_size, p=percentage
    )

    # Retreive model
    model = get_model_by_tag(args.net, device)

    # logger.info("Number of samples: {} ({}%)".format(n_samples, p))
    logger.info("Number of paramters: %s", count_params(model))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    data = []
    # Run epochs
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        # Run train
        train(model, device, train_loader, optimizer, epoch, args.log_interval)
        # Evaluate model on train/test data
        # train_loss, train_acc = evaluate_model(model, device, train_loader, "Train")
        # test_loss, test_acc = evaluate_model(model, device, test_loader, "Test")
        # data.append([epoch, train_acc, test_acc, train_loss, test_loss])

    # column_names = ["epoch", "train_acc", "test_acc", "train_loss", "test_loss"]
    # store_results(
    #     result_dir=os.path.join(args.result_dir, args.experiment_name),
    #     dataset_name="mnist-p={0:.2f}".format(percentage),
    #     column_names=column_names,
    #     data=data,
    # )

    # Evaluate model on train/test data
    train_loss, train_acc = evaluate_model(model, device, train_loader, "Train")
    test_loss, test_acc = evaluate_model(model, device, test_loader, "Test")
    return percentage, train_acc, test_acc, train_loss, test_loss


class FewShotExperiment:
    """Few shot experiment class."""

    def __init__(self, args):
        """
        Initialize the experiment.

        Args:
            args: Experiment options.
        """
        self.args = args

    def run(self):
        """Run the few shot experiment."""

        min_percentage = 0.05  # Equals 3 images per class
        max_percentage = 2.5
        percentage_step_size = 0.1

        if self.args.debug:
            max_percentage = 0.2

        processes = []
        pool = mp.Pool(processes=self.args.njobs)
        results = []

        # Iterate over percentages in  steps
        for p in np.arange(
            min_percentage, max_percentage + min_percentage, percentage_step_size
        ):
            try:

                r = pool.apply_async(run_for_percentage, args=[p, self.args])
                results.append(r)

            except Exception as e:
                logger.error("Exception at percentage {}:".format(p))
                logger.error(e)
                logger.error("--------------")

        pool.close()
        pool.join()

        logger.info("Joined all processes")

        rs = []
        # Get all results
        for res in results:
            try:
                r = res.get()
                rs.append(r)
            except Exception as e:
                logger.exception("Could not get result.")

        logger.info("Results %s", results)

        results = sorted(rs, key=lambda t: t[0])

        # Store results
        column_names = ["p", "train_acc", "test_acc", "train_loss", "test_loss"]
        data = np.array(results)
        store_results(
            result_dir=os.path.join(self.args.result_dir, self.args.experiment_name),
            dataset_name="mnist",
            column_names=column_names,
            data=data,
        )


def main_fewshot():
    """Run the MNIST experiment."""
    log_file = os.path.join(ARGS.result_dir, ARGS.experiment_name, "log.txt")
    print("Result dir: %s", ARGS.result_dir)
    print("Log file: %s", log_file)

    # Setup logging in base_dir/log.txt
    setup_logging(level=ARGS.log_level, filename=log_file)
    logger.info(" -- MNIST Few Shot Experiment -- Started ")
    tstart = time.time()

    try:
        if not ARGS.cuda:
            # Set number of CPU threads
            torch.set_num_threads(1)

        # Create and run experiment
        experiment = FewShotExperiment(ARGS)
        experiment.run()
    except Exception as e:
        logger.exception("Experiment crashed.")
        logger.exception("Exception: %s", str(e))

    # Measure time
    tstr = time_delta_now(tstart)
    logger.info(" -- MNIST -- Finished, took %s", tstr)


if __name__ == "__main__":
    ARGS = parse_args()
    main_fewshot()
