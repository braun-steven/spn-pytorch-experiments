import logging
import os
import sys
import time
import argparse
import multiprocessing as mp

import numpy as np
import torch

from src.utils.args import save_args
from src.utils.args import init_argparser
from src.utils.args import clone_args
from src.utils.utils import set_cuda_device, setup_logging, time_delta_now
from src.utils.utils import generate_run_base_dir
from src.utils.utils import generate_experiment_dir
from src.models import main_multilabel, main_cifar

logger = logging.getLogger(__name__)

mains = {
    "multilabel-mnist": main_multilabel.run_multilabel_mnist,
    "cifar": main_cifar.run_cifar,
}
get_args = {
    "multilabel-mnist": main_multilabel.parse_args,
    "cifar": main_cifar.parse_args,
}


def run(main_method, exp_dir, args, cuda_queue):
    """Run the MNIST experiment."""
    # Get cuda device from multiplrocessing queue
    cuda_device_id = cuda_queue.get()
    args.cuda_device_id = cuda_device_id
    print(
        "Starting {} with args \n{}\non device {}.".format(
            main_method.__name__, args, cuda_device_id
        )
    )
    print("os.environ[CUDA_VISIBLE_DEVICES]=", os.environ["CUDA_VISIBLE_DEVICES"])
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={"float_kind": float_formatter})

    # Setup logging in exp_dir/log.txt
    log_file = os.path.join(exp_dir, "log.txt")
    setup_logging(level=args.log_level, filename=log_file)
    logger.info(" -- MNIST Multilabel -- Started ")
    print("Result dir: ", args.result_dir)
    print("Base dir: ", exp_dir)
    print("Log file: ", log_file)

    # Save commandline arguments
    save_args(args, exp_dir)

    tstart = time.time()
    try:
        # Set number of CPU threads
        torch.set_num_threads(args.njobs)

        # Create and run experiment
        main_method(args, exp_dir)
    except Exception as e:
        logger.exception("Experiment crashed.")
        logger.exception("Exception: %s", str(e))

    # Measure time
    tstr = time_delta_now(tstart)
    logger.info(" -- MNIST -- Finished, took %s", tstr)

    # Free up cuda device
    cuda_queue.put(cuda_device_id)


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def main(exp):
    # Retrieve the experiment method and the arguments
    main_method = mains[exp]
    args_method = get_args[exp]
    args = args_method()

    # Now arguments can be modified and the experiment can be started with multiple different
    # arguments

    # Create a queue for all available cuda devices
    set_cuda_device(args.cuda_device_id)
    cuda_queue = mp.Queue(maxsize=len(args.cuda_device_id))
    processes = []
    # Add all cuda devices
    for dev in range(len(args.cuda_device_id)):
        cuda_queue.put(dev)

    suffix = "gridsearch"
    timestamp = time.time()
    for net in ["resnet+spn"]:
        for batch_size in [64, 128, 256]:
            for lr in [0.001, 0.0001]:
                for l2 in [0, 0.1, 0.01, 0.001, 0.0001]:
                    # Clone arguments
                    tmp_args = clone_args(args)

                    ####################
                    # Modify Arguments #
                    ####################
                    tmp_args.experiment_name = net
                    tmp_args.batch_size = batch_size
                    tmp_args.test_batch_size = batch_size
                    tmp_args.lr = lr
                    tmp_args.l2 = l2
                    tmp_args.net = net

                    # Generate tag for directory
                    tag = f"bs={batch_size}_lr={lr}_l2={l2}"

                    # Check if the base dir shall be reused
                    if args.reuse_base_dir is not None:
                        base_dir = args.reuse_base_dir
                    else:
                        base_dir = generate_run_base_dir(
                            suffix=suffix,
                            experiment=exp,
                            result_dir=tmp_args.result_dir,
                            timestamp=timestamp,
                        )

                    # Generate the experiment directory
                    exp_dir = generate_experiment_dir(base_dir, net, tag)

                    # Create process for this experiment
                    p = mp.Process(
                        target=run, args=(main_method, exp_dir, tmp_args, cuda_queue)
                    )
                    processes.append(p)
                    print(f"Process with tag={tag}, net={net}")

    print(f"Process list size: {len(processes)}")
    print("Starting all processes.")

    # Start processes
    for p in processes:
        p.start()

    # Join processes
    for p in processes:
        p.join()
    print("All processes finished.")


if __name__ == "__main__":
    # Find correct experiment
    if "--experiment" in sys.argv:
        idx = sys.argv.index("--experiment")
        exp = sys.argv[idx + 1]
        mp.set_start_method("spawn")
        main(exp)
    else:
        raise ValueError("No --experiment option was set.")
