import sys
import traceback

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
import os

import numpy as np
import torch
from torch import optim
from src.utils.utils import count_params

from src.data.data import store_results
from src.data.data_loader import get_mnist_loaders
from src.data.data_loader import get_mnist_subset
from src.models.mnist import evaluate_model, train
from src.models.models import get_model_by_tag
from torch.utils.data import SubsetRandomSampler

logger = logging.getLogger(__name__)


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
        use_cuda = self.args.cuda and torch.cuda.is_available()
        torch.manual_seed(self.args.seed)
        device = torch.device("cuda" if use_cuda else "cpu")

        min_percentage = 0.05  # Equals 3 images per class
        max_percentage = 5
        percentage_step_size = 0.1

        # Collect acc and loss
        train_accs, test_accs = [], []
        train_losses, test_losses = [], []

        # Percentages
        ps = []

        if ARGS.debug:
            max_percentage = 3

        try:

            # Iterate over percentages in  steps
            for p in np.arange(
                min_percentage, max_percentage + min_percentage, percentage_step_size
            ):
                # Set seed globally
                set_seed(ARGS.seed)

                ARGS.batch_size = int(60000 * p / 100 * 1 / 10)
                logger.info(
                    "Current percentage: %.2f, Batch size: %s", p, ARGS.batch_size
                )

                # Get the mnist loader
                train_loader, test_loader = get_mnist_subset(
                    use_cuda=use_cuda, args=self.args, p=p
                )

                # Retreive model
                model = get_model_by_tag(self.args.net, device)

                # logger.info("Number of samples: {} ({}%)".format(n_samples, p))
                logger.info("Number of paramters: %s", count_params(model))

                # Define optimizer
                optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

                # Scheduler for learning rate
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=25, gamma=0.5
                )

                # Run epochs
                for epoch in range(1, self.args.epochs + 1):
                    scheduler.step()

                    # Run train
                    train(
                        model,
                        device,
                        train_loader,
                        optimizer,
                        epoch,
                        self.args.log_interval,
                    )

                # Evaluate model on train/test data
                train_loss, train_acc = evaluate_model(
                    model, device, train_loader, "Train"
                )
                test_loss, test_acc = evaluate_model(model, device, test_loader, "Test")

                # Store acc/loss
                train_accs.append(train_acc)
                train_losses.append(train_loss)
                test_accs.append(test_acc)
                test_losses.append(test_loss)
                ps.append(p)

        except Exception as e:
            logger.info("Exception:")
            traceback.print_exc()
            logger.info("--------------")
            logger.info("Saving results")

        # Store results
        column_names = ["p", "train_acc", "test_acc", "train_loss", "test_loss"]
        data = np.c_[ps, train_accs, test_accs, train_losses, test_losses]
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
            torch.set_num_threads(ARGS.njobs)

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
