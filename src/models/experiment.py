import logging
import os

import numpy as np
import torch
from torch import optim
from src.utils.utils import count_params

from src.data.data import store_results
from src.data.data_loader import get_mnist_loaders
from src.models.mnist import evaluate_model, train
from src.models.models import get_model_by_tag

logger = logging.getLogger(__name__)


class MnistExperiment:
    """Main experiment class."""

    def __init__(self, args):
        """
        Initialize the experiment.

        Args:
            args: Experiment options.
        """
        self.args = args

    def run(self):
        """Run the MNIST experiment."""
        use_cuda = self.args.cuda and torch.cuda.is_available()
        torch.manual_seed(self.args.seed)
        device = torch.device("cuda" if use_cuda else "cpu")

        # Get the mnist loader
        train_loader, test_loader = get_mnist_loaders(use_cuda, self.args)

        model = get_model_by_tag(self.args.net, device)

        # with SummaryWriter(comment="Model", log_dir="tensorboard") as w:
        #     w.add_graph(model, torch.zeros(1, 28, 28), True)
        # exit()

        logger.info("Number of paramters: %s", count_params(model))
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        # Scheduler for learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

        # Collect acc and loss
        train_accs, test_accs = [], []
        train_losses, test_losses = [], []

        # Run epochs
        for epoch in range(1, self.args.epochs + 1):
            scheduler.step()

            # Run train
            train(model, device, train_loader, optimizer, epoch, self.args.log_interval)

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
            result_dir=os.path.join(self.args.result_dir, self.args.experiment_name),
            dataset_name="mnist",
            column_names=column_names,
            data=data,
        )
