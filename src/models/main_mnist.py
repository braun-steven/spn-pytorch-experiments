"""
Main experiment on MNIST.

Compare MLP vs SPN-MLP.
"""
import sys
import traceback

sys.path.insert(0, "./")
import logging
import os
import time

from src.utils.args import parse_args
from src.utils.utils import ensure_dir, setup_logging, time_delta_now
from src.models.experiment import MnistExperiment

# Logger instance
logger = logging.getLogger(__name__)


def main_mnist():
    """Run the MNIST experiment."""
    log_file = os.path.join(ARGS.result_dir, ARGS.experiment_name, "log.txt")
    print("Result dir: %s", ARGS.result_dir)
    print("Log file: %s", log_file)

    # Setup logging in base_dir/log.txt
    setup_logging(level=ARGS.log_level, filename=log_file)
    logger.info(" -- MNIST -- Started ")
    tstart = time.time()

    try:
        # Create and run experiment
        experiment = MnistExperiment(ARGS)
        experiment.run()
    except Exception as e:
        logger.exception("Experiment crashed.")
        logger.exception("Exception: %s", str(e))

    # Measure time
    tstr = time_delta_now(tstart)
    logger.info(" -- MNIST -- Finished, took %s", tstr)


if __name__ == "__main__":
    ARGS = parse_args()
    main_mnist()
