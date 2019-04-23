import random
import numpy as np
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
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"


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


def generate_run_base_dir(tag: str, result_dir: str) -> str:
    """
    Generate a base directory for each experiment run.
    Args:
        tag (str): Experiment tag.
        result_dir (str): Experiment output directory.

    Returns:
        str: Directory name.
    """
    _date_str = datetime.datetime.today().strftime("%y-%m-%d_%Hh:%Mm")
    tagstr = tag if tag == "" else "_" + tag
    base_dir = os.path.join(result_dir, f"{_date_str}{tagstr}")
    os.makedirs(base_dir)
    return base_dir


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
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = torch.Tensor(target)

    return target
