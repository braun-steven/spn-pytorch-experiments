from typing import List
from src.utils.utils import ensure_dir
import numpy as np
import os


def store_results(result_dir, dataset_name, column_names, data):
    """
    Store results.

    Args:
        result_dir (str): Result directory.
        dataset_name (str): Dataset name.
        column_names (List[str]): List of column names
        data (np.ndarray): Result data which is to be stored.
    """
    # Write header and content into a csv file
    fname = "./{}/{}.csv".format(result_dir, dataset_name)
    ensure_dir(os.path.dirname(fname))
    np.savetxt(fname, data, delimiter=",", header=",".join(column_names))
