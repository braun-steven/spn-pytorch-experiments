from typing import List
from src.utils.utils import ensure_dir
import numpy as np
import os


def store_results(result_dir, dataset_name, exp_name, column_names, data):
    """
    Store results.

    Args:
        result_dir (str): Result directory.
        dataset_name (str): Dataset name.
        exp_name (str): Experiment name.
        column_names (List[str]): List of column names
        data (np.ndarray): Result data which is to be stored.
    """
    results_dir = result_dir
    exp_dir = os.path.join(results_dir, exp_name)
    ensure_dir(exp_dir)
    # Write header and content into a csv file
    fname = "./{}/{}.csv".format(exp_dir, dataset_name)
    np.savetxt(fname, data, delimiter=",", header=",".join(column_names))
