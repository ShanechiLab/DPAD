""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""Tensorflow tools"""

import logging
import os
import time

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def set_global_tf_eagerly_flag(desired_flag=False):
    global_tf_eagerly_flag = (
        tf.config.functions_run_eagerly()
    )  # Get global eager execution config
    if global_tf_eagerly_flag != desired_flag:
        logger.info(
            f"Changing global Tensorflow eagerly flag from {global_tf_eagerly_flag} to {desired_flag}"
        )
        tf.config.run_functions_eagerly(desired_flag)  # Disable global eager execution
    return global_tf_eagerly_flag


def setupTensorflow(cpu=False):
    logger.info("Tensorflow version: {}".format(tf.__version__))
    logger.info("Tensorflow path: {}".format(os.path.abspath(tf.__file__)))
    if cpu == False:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices("GPU")
                logger.info(
                    f"Found {len(logical_gpus)} Logical GPU(s) and {len(gpus)} Physical GPU(s): {gpus}"
                )
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                logger.info(e)
        else:
            logger.info("No GPUs were found!")
    else:
        cpus = tf.config.list_physical_devices("CPU")
        logger.info("Using CPUs: {}".format(cpus))
        pass


def convertHistoryToDict(history, tic=None):
    """Converts tf model.fit history to a dictionary.

    Args:
        history (model.fit output): output of model.fit for a tf model.

    Returns:
        dict: dictionary form of the history.
    """
    if tic is not None:
        toc = time.perf_counter()
        fit_time = toc - tic
    else:
        fit_time = None
    return {
        "epoch": history.epoch,
        "history": history.history,
        "params": history.params,
        "fit_time": fit_time,
    }


def getModelFitHistoyStr(
    history=None, fields=["loss"], keep_ratio=1, history_dict=None, epoch=None
):
    """Prints a human readable summary of a tf model.fit history

    Args:
        history (model.fit output): output of model.fit for a tf model. Defaults to None.
        fields (list, optional): fields to print. Defaults to ['loss'].
        keep_ratio (int, optional): ratio of epochs to include in the log. Defaults to 1.
        history_dict (dict, optional): dictionary form of the history. Defaults to None.
        epoch (int, optional): number of epochs. Defaults to None.
    """
    if epoch is None:
        epoch = history.epoch
    if history_dict is None:
        history_dict = history.history
    if keep_ratio < 1 and len(epoch) > 0:
        epochToPrint = list(range(0, len(epoch), int(np.ceil(keep_ratio * len(epoch)))))
        if (len(epoch) - 1) not in epochToPrint:
            epochToPrint.append(len(epoch) - 1)
    else:
        epochToPrint = range(len(epoch))
    logStrAll = ""
    for ei in epochToPrint:
        logStr = "Epoch {}/{} - ".format(1 + epoch[ei], len(epoch))
        metricNameStrs = []
        metricVals = []
        for f in fields:
            val = history_dict[f][ei]
            if val not in metricVals:
                metricVals.append(val)
                metricNameStrs.append(f)
            else:
                ind = metricVals.index(val)
                metricNameStrs[ind] += "={}".format(f)
        metricStrs = [
            "{}={:.8g}".format(mName, mVal)
            for mName, mVal in zip(metricNameStrs, metricVals)
        ]
        logStr += ", ".join(metricStrs)
        logStrAll += logStr + "\n"
    return logStrAll
