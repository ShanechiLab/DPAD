""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

import bz2
import gzip
import os
import pickle
from collections import deque
from itertools import chain
from pathlib import Path
from sys import stderr

try:
    from reprlib import repr
except ImportError:
    pass

import numpy as np


def pickle_load(filePath):
    """Loads a pickle file

    Args:
        filePath (string): file path

    Returns:
        data (Any): pickle file content
    """
    with open(filePath, "rb") as f:
        return pickle.load(f)


def pickle_save(filePath, data):
    """Saves a pickle file

    Args:
        filePath (string): file path
        data (Any): data to save in file
    """
    with open(filePath, "wb") as f:
        pickle.dump(data, f)


def pickle_load_compressed(filePath, format="bz2", auto_add_ext=False):
    """Loads data from a compressed pickle file

    Args:
        filePath (string): file path
        format (str, optional): the compression format, can be in ['bz2', 'gz']. Defaults to 'bz2'.
        auto_add_ext (bool, optional): if true, will automatically add the
            extension for the compression format to the file path. Defaults to False.

    Returns:
        data (Any): pickle file content
    """
    if format == "bz2":
        if auto_add_ext:
            filePath += ".bz2"
        data = bz2.BZ2File(filePath, "rb")
    elif format == "gzip":
        if auto_add_ext:
            filePath += ".gz"
        data = gzip.open(filePath, "rb")
    else:
        raise (Exception("Unsupported format: {}".format(format)))
    return pickle.load(data)


def pickle_save_compressed(filePath, data, format="bz2", auto_add_ext=False):
    """Saves data as a compressed pickle file

    Args:
        filePath (string): file path
        data (Any): data to save in file
        format (str, optional): the compression format, can be in ['bz2', 'gz']. Defaults to 'bz2'.
        auto_add_ext (bool, optional): if true, will automatically add the
            extension for the compression format to the file path. Defaults to False.
    """
    if format == "bz2":
        if auto_add_ext:
            filePath += ".bz2"
        with bz2.BZ2File(filePath, "w") as f:
            pickle.dump(data, f)
    elif format == "gzip":
        if auto_add_ext:
            filePath += ".gz"
        with gzip.open(filePath, "w") as f:
            pickle.dump(data, f)
    else:
        raise (Exception("Unsupported format: {}".format(format)))


def mk_parent_dir(file_path):
    return Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)


def bytes_to_string(num, suffix="B"):
    """Gets size in bytes and returns a human readable string

    Args:
        num (number): input size in bytes
        suffix (str, optional): Suffix to add to the final string. Defaults to 'B'.

    Returns:
        string: human readable string of the input size
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def file_exists(filePath):
    """Returns true if file exists

    Args:
        filePath (string): file path

    Returns:
        output (bool): True if file exists
    """
    return os.path.exists(filePath)


def get_file_size(filePath):
    """Returns the file size in bytes

    Args:
        filePath (string): file path

    Returns:
        size (number): size of file in bytes
    """
    return os.path.getsize(filePath)
