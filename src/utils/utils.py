import numpy as np
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
