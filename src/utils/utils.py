import random
from src.spn import distributions as spndist
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import os
import datetime
import logging
import sys


def time_delta_now(ts: float) -> str:
    """
    Convert a timestamp into a human readable timestring (%H:%M:%S).
    Args:
        ts (float): Timestamp.

    Returns:
        Human readable timestring.
    """
    a = ts
    b = time.time()  # current epoch time
    c = b - a  # seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds"


def ensure_dir(d):
    """Ensure that a directory exists"""
    # Create result dir on the fly
    if not os.path.exists(d):
        os.makedirs(d)


def count_params(model) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_run_base_dir(
    experiment: str, suffix: str, result_dir: str, timestamp
) -> str:
    """
    Generate a base directory for each experiment run.
    Looks like this: {result_dir}/{experiment}/{date}_{suffix}/{tag}/{arch}
    Args:
        result_dir (str): Experiment output directory.

    Returns:
        str: Directory name.
    """
    date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%y%m%d_%H%M")
    base_dir = os.path.join(result_dir, experiment, f"{date_str}_{suffix}")
    ensure_dir(base_dir)
    return base_dir


def generate_experiment_dir(base_dir: str, arch: str, tag: str) -> str:
    """
    Generate a base directory for each experiment run.

    Returns:
        str: Directory name.
    """
    exp_dir = os.path.join(base_dir, tag, arch)
    ensure_dir(exp_dir)
    return exp_dir


def setup_logging(filename: str = "log.txt", level: str = "INFO"):
    """
    Setup global loggers.

    Args:
        filename: Log file destination.
        level: Log level.
    """
    # Make sure the directory actually exists
    ensure_dir(os.path.dirname(filename))

    # Check if previous log exists since logging.FileHandler only appends
    if os.path.exists(filename):
        os.remove(filename)

    logging.basicConfig(
        level=logging.getLevelName(level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(filename=filename),
        ],
    )


def set_seed(seed: int):
    """
    Set the seed globally for cuda, numpy and torch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_one_hot(labels, C=10):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    """
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(1)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = torch.Tensor(target)

    return target


def collect_tensorboard_info(
    writer: SummaryWriter,
    model: nn.Module,
    epoch: int,
    train_acc: float,
    test_acc: float,
    train_loss: float,
    test_loss: float,
):
    # Fill tensorboard
    writer.add_scalar(tag="accuracy/train", scalar_value=train_acc, global_step=epoch)
    writer.add_scalar(tag="accuracy/test", scalar_value=test_acc, global_step=epoch)
    writer.add_scalar(tag="loss/train", scalar_value=train_loss, global_step=epoch)
    writer.add_scalar(tag="loss/test", scalar_value=test_loss, global_step=epoch)

    # Add sum weight histograms
    for name, module in model.named_modules():
        if hasattr(module, "sum_weights"):
            ws = module.sum_weights.detach()
            tag = "{}_sum_weights".format(name)
            writer.add_histogram(tag=tag, values=ws.view(-1), global_step=epoch)

        # Collect means
        if hasattr(module, "means") and type(module) == spndist.Normal:
            # Change of means over epoch
            # for i, mean in enumerate(module.means.view(-1)):
            #     tag = "{}_mean_{}".format(name, i)
            #     writer.add_scalar(tag, mean, epoch)

            # Mean distribution
            writer.add_histogram(
                tag="{}_mean_dist".format(name),
                values=module.means.view(-1),
                global_step=epoch,
            )

        # Collect standard deviations
        if hasattr(module, "stds") and type(module) == spndist.Normal:
            # for i, std in enumerate(module.stds.view(-1)):
            #     tag = "{}_std_{}".format(name, i)
            #     writer.add_scalar(tag, std, epoch)
            # Std distribution
            writer.add_histogram(
                tag="{}_std_dist".format(name),
                values=module.stds.view(-1),
                global_step=epoch,
            )


def get_n_samples_from_loader(loader) -> int:
    """
    Get the number of samples in the data loader.
    Respects if the data loader has a sampler.
    Args:
        loader: Data loader.
    Returns:
        int: Number of samples in that data loader.
    """
    n_samples = len(loader.dataset)

    # If sampler is set, use the size of the sampler
    if loader.sampler:
        n_samples = len(loader.sampler)

    return n_samples


def set_cuda_device(cuda_device_id):
    """
    Set the visible cuda devices.

    Args:
        cuda_device_id (List[int]): Cuda device ids.

    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cuda_device_id])
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
